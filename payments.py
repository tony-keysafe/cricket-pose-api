"""
Stripe payment integration for Cricket Analyze Pro.

Environment variables required (set in Railway dashboard):
  STRIPE_SECRET_KEY         - sk_test_... or sk_live_...
  STRIPE_WEBHOOK_SECRET     - whsec_... (from Stripe webhook settings)
  STRIPE_PRICE_PRO          - price_... (Pro monthly price ID from Stripe dashboard)
  STRIPE_PRICE_TEAM         - price_... (Team monthly price ID from Stripe dashboard)
  FRONTEND_URL              - https://cricket-analyze-pro.vercel.app

Setup steps:
  1. Create Stripe account at stripe.com
  2. Create two Products in Stripe dashboard:
     - "Pro" (£9.99/month recurring) → copy price ID → set STRIPE_PRICE_PRO
     - "Team" (£29.99/month recurring) → copy price ID → set STRIPE_PRICE_TEAM
  3. Add webhook endpoint in Stripe dashboard:
     URL: https://cricket-pose-api-production.up.railway.app/webhook/stripe
     Events: checkout.session.completed, customer.subscription.updated,
             customer.subscription.deleted, invoice.payment_failed
  4. Copy webhook signing secret → set STRIPE_WEBHOOK_SECRET
  5. Copy secret key → set STRIPE_SECRET_KEY
  6. Set FRONTEND_URL=https://cricket-analyze-pro.vercel.app
"""

import os
import json
import time
import logging
from pathlib import Path
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger("stripe_payments")

router = APIRouter()

# ─── Config ──────────────────────────────────────────────────────
STRIPE_SECRET_KEY = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PRO = os.environ.get("STRIPE_PRICE_PRO", "")
STRIPE_PRICE_TEAM = os.environ.get("STRIPE_PRICE_TEAM", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "https://cricket-analyze-pro.vercel.app")

# Simple JSON file-based customer store (swap for database later)
CUSTOMERS_FILE = Path("/app/customers.json") if os.path.isdir("/app") else Path("customers.json")

stripe = None

def init_stripe():
    """Initialize Stripe SDK. Call on startup."""
    global stripe
    if not STRIPE_SECRET_KEY:
        logger.warning("STRIPE_SECRET_KEY not set — payment endpoints will return 503")
        return False
    try:
        import stripe as _stripe
        _stripe.api_key = STRIPE_SECRET_KEY
        stripe = _stripe
        logger.info("Stripe initialized successfully")
        return True
    except ImportError:
        logger.error("stripe package not installed — run: pip install stripe")
        return False

def _require_stripe():
    """Raise 503 if Stripe isn't configured."""
    if stripe is None:
        raise HTTPException(status_code=503, detail="Payment system not configured. Set STRIPE_SECRET_KEY.")

# ─── Customer Store (simple JSON — replace with DB) ──────────────
def _load_customers():
    if CUSTOMERS_FILE.exists():
        try:
            return json.loads(CUSTOMERS_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_customers(data):
    CUSTOMERS_FILE.write_text(json.dumps(data, indent=2))

def _get_customer(email):
    customers = _load_customers()
    return customers.get(email.lower().strip())

def _set_customer(email, data):
    customers = _load_customers()
    customers[email.lower().strip()] = {
        **customers.get(email.lower().strip(), {}),
        **data,
        "updated_at": int(time.time()),
    }
    _save_customers(customers)

# ─── Endpoints ───────────────────────────────────────────────────

@router.post("/checkout/create")
async def create_checkout(request: Request):
    """
    Create a Stripe Checkout session for subscription purchase.

    Body JSON:
      { "email": "user@example.com", "plan": "pro" | "team" }

    Returns:
      { "checkout_url": "https://checkout.stripe.com/..." }
    """
    _require_stripe()
    body = await request.json()
    email = body.get("email", "").strip().lower()
    plan = body.get("plan", "pro").lower()

    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")

    price_id = STRIPE_PRICE_PRO if plan == "pro" else STRIPE_PRICE_TEAM
    if not price_id:
        raise HTTPException(status_code=503, detail=f"Price ID not configured for plan: {plan}")

    # Find or create Stripe customer
    customer_data = _get_customer(email)
    customer_id = customer_data.get("stripe_customer_id") if customer_data else None

    if not customer_id:
        customer = stripe.Customer.create(email=email)
        customer_id = customer.id
        _set_customer(email, {
            "stripe_customer_id": customer_id,
            "plan": "free",
            "status": "inactive",
        })

    # Create checkout session
    session = stripe.checkout.Session.create(
        customer=customer_id,
        payment_method_types=["card"],
        line_items=[{
            "price": price_id,
            "quantity": 1,
        }],
        mode="subscription",
        success_url=f"{FRONTEND_URL}/settings?checkout=success",
        cancel_url=f"{FRONTEND_URL}/pricing?checkout=cancelled",
        metadata={
            "email": email,
            "plan": plan,
        },
    )

    return {"checkout_url": session.url}


@router.post("/portal/create")
async def create_portal(request: Request):
    """
    Create a Stripe Customer Portal session for managing subscription.

    Body JSON:
      { "email": "user@example.com" }

    Returns:
      { "portal_url": "https://billing.stripe.com/..." }
    """
    _require_stripe()
    body = await request.json()
    email = body.get("email", "").strip().lower()

    customer_data = _get_customer(email)
    if not customer_data or not customer_data.get("stripe_customer_id"):
        raise HTTPException(status_code=404, detail="No subscription found for this email")

    session = stripe.billing_portal.Session.create(
        customer=customer_data["stripe_customer_id"],
        return_url=f"{FRONTEND_URL}/settings",
    )

    return {"portal_url": session.url}


@router.get("/subscription/status")
async def subscription_status(request: Request):
    """
    Check subscription status for an email.

    Query params:
      ?email=user@example.com

    Returns:
      { "plan": "free"|"pro"|"team", "status": "active"|"inactive"|"past_due", "email": "..." }
    """
    _require_stripe()
    email = request.query_params.get("email", "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="Email required")

    customer_data = _get_customer(email)
    if not customer_data:
        return {
            "email": email,
            "plan": "free",
            "status": "inactive",
        }

    return {
        "email": email,
        "plan": customer_data.get("plan", "free"),
        "status": customer_data.get("status", "inactive"),
        "stripe_customer_id": customer_data.get("stripe_customer_id"),
    }


@router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    Verifies signature, processes subscription lifecycle events.
    """
    if stripe is None:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")

    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        else:
            # No webhook secret — parse without verification (dev only!)
            event = json.loads(payload)
            logger.warning("⚠️  Webhook signature verification SKIPPED (no STRIPE_WEBHOOK_SECRET)")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid webhook signature")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    event_type = event.get("type", "")
    data = event.get("data", {}).get("object", {})

    logger.info(f"Stripe webhook: {event_type}")

    if event_type == "checkout.session.completed":
        _handle_checkout_completed(data)
    elif event_type == "customer.subscription.updated":
        _handle_subscription_updated(data)
    elif event_type == "customer.subscription.deleted":
        _handle_subscription_deleted(data)
    elif event_type == "invoice.payment_failed":
        _handle_payment_failed(data)
    else:
        logger.info(f"Unhandled webhook event: {event_type}")

    return {"received": True}


# ─── Webhook Handlers ────────────────────────────────────────────

def _handle_checkout_completed(session):
    """User completed checkout — activate their subscription."""
    email = session.get("metadata", {}).get("email") or session.get("customer_email", "")
    plan = session.get("metadata", {}).get("plan", "pro")
    customer_id = session.get("customer")
    subscription_id = session.get("subscription")

    if email:
        _set_customer(email, {
            "stripe_customer_id": customer_id,
            "subscription_id": subscription_id,
            "plan": plan,
            "status": "active",
            "activated_at": int(time.time()),
        })
        logger.info(f"✅ Subscription activated: {email} → {plan}")


def _handle_subscription_updated(subscription):
    """Subscription changed (upgrade, downgrade, renewal)."""
    customer_id = subscription.get("customer")
    status = subscription.get("status")  # active, past_due, canceled, etc.

    # Find customer by stripe_customer_id
    customers = _load_customers()
    for email, data in customers.items():
        if data.get("stripe_customer_id") == customer_id:
            # Determine plan from price
            items = subscription.get("items", {}).get("data", [])
            price_id = items[0].get("price", {}).get("id", "") if items else ""

            plan = "pro"
            if price_id == STRIPE_PRICE_TEAM:
                plan = "team"
            elif price_id == STRIPE_PRICE_PRO:
                plan = "pro"

            _set_customer(email, {
                "plan": plan if status == "active" else data.get("plan", "free"),
                "status": status,
            })
            logger.info(f"🔄 Subscription updated: {email} → {plan} ({status})")
            break


def _handle_subscription_deleted(subscription):
    """Subscription cancelled or expired."""
    customer_id = subscription.get("customer")
    customers = _load_customers()
    for email, data in customers.items():
        if data.get("stripe_customer_id") == customer_id:
            _set_customer(email, {
                "plan": "free",
                "status": "cancelled",
                "cancelled_at": int(time.time()),
            })
            logger.info(f"❌ Subscription cancelled: {email}")
            break


def _handle_payment_failed(invoice):
    """Payment failed — mark as past_due."""
    customer_id = invoice.get("customer")
    customers = _load_customers()
    for email, data in customers.items():
        if data.get("stripe_customer_id") == customer_id:
            _set_customer(email, {"status": "past_due"})
            logger.info(f"⚠️  Payment failed: {email}")
            break
