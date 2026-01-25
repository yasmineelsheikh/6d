# Stripe Integration Setup

This document explains how to set up Stripe payment processing for the billing system.

## Environment Variables

Add the following environment variables to your `.env` file or deployment environment:

```bash
# Stripe Secret Key (from Stripe Dashboard)
STRIPE_SECRET_KEY=sk_test_...  # Use sk_live_... for production

# Stripe Publishable Key (from Stripe Dashboard)
STRIPE_PUBLISHABLE_KEY=pk_test_...  # Use pk_live_... for production

# Stripe Webhook Secret (from Stripe Dashboard after creating webhook)
STRIPE_WEBHOOK_SECRET=whsec_...  # Optional but recommended for production
```

## Getting Your Stripe Keys

1. **Create a Stripe Account**: Go to https://stripe.com and create an account
2. **Get API Keys**:
   - Go to https://dashboard.stripe.com/test/apikeys (for test mode)
   - Copy your "Secret key" → `STRIPE_SECRET_KEY`
   - Copy your "Publishable key" → `STRIPE_PUBLISHABLE_KEY`
3. **For Production**: Switch to "Live mode" and get live keys

## Setting Up Webhooks (Recommended)

Webhooks allow Stripe to notify your backend when payments succeed, ensuring credits are added even if the frontend confirmation fails.

1. **In Stripe Dashboard**: Go to Developers → Webhooks
2. **Add Endpoint**: 
   - URL: `https://your-backend-url.com/api/billing/webhook`
   - Events to listen for: `payment_intent.succeeded`
3. **Copy Webhook Secret**: After creating the webhook, copy the "Signing secret" → `STRIPE_WEBHOOK_SECRET`

## Testing

### Test Cards (Test Mode Only)

Use these test card numbers in Stripe test mode:

- **Success**: `4242 4242 4242 4242`
- **Decline**: `4000 0000 0000 0002`
- **Requires Authentication**: `4000 0025 0000 3155`

Use any future expiry date, any 3-digit CVC, and any ZIP code.

## Payment Flow

1. User enters number of credits to purchase
2. Frontend calls `/api/billing/purchase` to create a PaymentIntent
3. Frontend uses Stripe Elements to collect card details
4. Frontend confirms payment with Stripe
5. Frontend calls `/api/billing/confirm-payment` to add credits
6. (Optional) Stripe webhook also confirms payment and adds credits as backup

## Security Notes

- **Never expose your Secret Key** in frontend code
- Always use HTTPS in production
- Verify webhook signatures using `STRIPE_WEBHOOK_SECRET`
- The webhook endpoint should be publicly accessible (Stripe needs to call it)

## Troubleshooting

- **"Stripe is not configured"**: Check that `STRIPE_SECRET_KEY` is set
- **Payment fails**: Check Stripe Dashboard → Payments for error details
- **Credits not added**: Check webhook logs in Stripe Dashboard
- **CORS errors**: Ensure your backend allows requests from your frontend domain
