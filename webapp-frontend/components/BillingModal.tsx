'use client'

import { useState, useEffect } from 'react'
import { X } from 'lucide-react'
import { useAuth } from '@/contexts/AuthContext'
import { loadStripe } from '@stripe/stripe-js'
import { Elements, CardElement, useStripe, useElements } from '@stripe/react-stripe-js'

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || ''

interface BillingModalProps {
  isOpen: boolean
  onClose: () => void
}

// Payment form component that uses Stripe Elements
function PaymentForm({ creditsToPurchase, onSuccess, onError }: { 
  creditsToPurchase: number
  onSuccess: (credits: string) => void
  onError: (error: string) => void
}) {
  const stripe = useStripe()
  const elements = useElements()
  const { token } = useAuth()
  const [processing, setProcessing] = useState(false)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!stripe || !elements) {
      return
    }

    setProcessing(true)

    try {
      // Create payment intent
      const purchaseResponse = await fetch(`${API_BASE}/api/billing/purchase`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ credits: creditsToPurchase })
      })

      if (!purchaseResponse.ok) {
        const errorData = await purchaseResponse.json()
        throw new Error(errorData.detail || 'Failed to create payment intent')
      }

      const purchaseData = await purchaseResponse.json()
      const { client_secret, payment_intent_id } = purchaseData

      if (!client_secret) {
        throw new Error('No client secret received from server')
      }

      // Confirm payment with Stripe
      const cardElement = elements.getElement(CardElement)
      if (!cardElement) {
        throw new Error('Card element not found')
      }

      const { error: confirmError, paymentIntent } = await stripe.confirmCardPayment(
        client_secret,
        {
          payment_method: {
            card: cardElement,
          }
        }
      )

      if (confirmError) {
        throw new Error(confirmError.message || 'Payment failed')
      }

      if (paymentIntent?.status === 'succeeded') {
        // Confirm with backend
        const confirmResponse = await fetch(`${API_BASE}/api/billing/confirm-payment`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({ payment_intent_id: payment_intent_id || paymentIntent.id })
        })

        if (!confirmResponse.ok) {
          const errorData = await confirmResponse.json()
          throw new Error(errorData.detail || 'Failed to confirm payment')
        }

        const confirmData = await confirmResponse.json()
        onSuccess(confirmData.credits)
      } else {
        throw new Error(`Payment status: ${paymentIntent?.status}`)
      }
    } catch (err: any) {
      onError(err.message || 'Payment failed')
    } finally {
      setProcessing(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="p-3 bg-[#1a1a1a] border border-[#2a2a2a] rounded">
        <CardElement
          options={{
            style: {
              base: {
                fontSize: '14px',
                color: '#d4d4d4',
                '::placeholder': {
                  color: '#666666',
                },
              },
              invalid: {
                color: '#ef4444',
              },
            },
          }}
        />
      </div>
      <button
        type="submit"
        disabled={!stripe || processing}
        className="w-full px-4 py-2 bg-[#3a3a3a] hover:bg-[#4a4a4a] text-[#d4d4d4] text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
      >
        {processing ? 'Processing...' : `Pay $${(creditsToPurchase * 0.1).toFixed(2)}`}
      </button>
    </form>
  )
}

export default function BillingModal({ isOpen, onClose }: BillingModalProps) {
  const { user, token } = useAuth()
  const [credits, setCredits] = useState<string>('0')
  const [purchaseAmount, setPurchaseAmount] = useState<string>('100')
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [stripePromise, setStripePromise] = useState<Promise<any> | null>(null)
  const [showPaymentForm, setShowPaymentForm] = useState(false)

  useEffect(() => {
    if (isOpen && user) {
      setCredits(user.credits || '0')
      loadCredits()
      loadStripeKey()
    }
  }, [isOpen, user])

  const loadStripeKey = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/billing/stripe-key`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      if (response.ok) {
        const data = await response.json()
        if (data.publishable_key) {
          setStripePromise(loadStripe(data.publishable_key))
        }
      }
    } catch (err) {
      console.error('Error loading Stripe key:', err)
    }
  }

  const loadCredits = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/billing/credits`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      if (response.ok) {
        const data = await response.json()
        setCredits(data.credits || '0')
      }
    } catch (err) {
      console.error('Error loading credits:', err)
    }
  }

  const handlePurchaseClick = () => {
    const creditsToPurchase = parseInt(purchaseAmount)
    if (isNaN(creditsToPurchase) || creditsToPurchase <= 0) {
      setError('Please enter a valid number of credits')
      return
    }
    setError(null)
    setSuccess(null)
    setShowPaymentForm(true)
  }

  const handlePaymentSuccess = (newCredits: string) => {
    setCredits(newCredits)
    const creditsToPurchase = parseInt(purchaseAmount)
    const cost = (creditsToPurchase * 0.1).toFixed(2)
    setSuccess(`Successfully purchased ${creditsToPurchase} credits for $${cost}`)
    setPurchaseAmount('100')
    setShowPaymentForm(false)
    
    // Update user in context
    if (user) {
      user.credits = newCredits
    }
  }

  const handlePaymentError = (errorMessage: string) => {
    setError(errorMessage)
    setShowPaymentForm(false)
  }

  if (!isOpen) return null

  const creditsNum = parseInt(credits) || 0
  const costPerCredit = 0.1
  const purchaseCreditsNum = parseInt(purchaseAmount) || 0
  const totalCost = (purchaseCreditsNum * costPerCredit).toFixed(2)

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-[#222222] border border-[#2a2a2a] rounded-lg w-full max-w-md">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-[#2a2a2a]">
          <h2 className="text-lg font-semibold text-[#d4d4d4]">Billing & Credits</h2>
          <button
            onClick={onClose}
            className="p-1 text-[#8a8a8a] hover:text-[#d4d4d4] transition-colors"
            aria-label="Close"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Current Credits */}
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4 rounded">
            <div className="text-xs text-[#8a8a8a] mb-1">Current Credits</div>
            <div className="text-2xl font-semibold text-[#d4d4d4]">{creditsNum.toLocaleString()}</div>
            <div className="text-xs text-[#8a8a8a] mt-2">
              1 credit = 1 second of generated video
            </div>
          </div>

          {/* Purchase Credits */}
          <div className="space-y-4">
            {!showPaymentForm ? (
              <>
                <div>
                  <label className="block text-sm font-medium text-[#d4d4d4] mb-2">
                    Purchase Credits
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="number"
                      min="1"
                      value={purchaseAmount}
                      onChange={(e) => setPurchaseAmount(e.target.value)}
                      className="flex-1 px-3 py-2 bg-[#1a1a1a] border border-[#2a2a2a] text-[#d4d4d4] placeholder:text-[#666666] text-sm focus:outline-none focus:border-[#3a3a3a] transition-colors"
                      placeholder="Number of credits"
                    />
                    <button
                      onClick={handlePurchaseClick}
                      className="px-4 py-2 bg-[#3a3a3a] hover:bg-[#4a4a4a] text-[#d4d4d4] text-sm font-medium transition-colors"
                    >
                      Continue
                    </button>
                  </div>
                  {purchaseCreditsNum > 0 && (
                    <div className="mt-2 text-xs text-[#8a8a8a]">
                      Cost: ${totalCost} (${costPerCredit.toFixed(2)} per credit)
                    </div>
                  )}
                </div>
              </>
            ) : (
              <>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-[#d4d4d4]">
                    Purchasing {purchaseCreditsNum} credits for ${totalCost}
                  </span>
                  <button
                    onClick={() => {
                      setShowPaymentForm(false)
                      setError(null)
                    }}
                    className="text-xs text-[#8a8a8a] hover:text-[#d4d4d4]"
                  >
                    Cancel
                  </button>
                </div>
                {stripePromise ? (
                  <Elements stripe={stripePromise}>
                    <PaymentForm
                      creditsToPurchase={purchaseCreditsNum}
                      onSuccess={handlePaymentSuccess}
                      onError={handlePaymentError}
                    />
                  </Elements>
                ) : (
                  <div className="text-xs text-[#8a8a8a]">Loading payment form...</div>
                )}
              </>
            )}

            {/* Error/Success Messages */}
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 text-xs p-3 rounded">
                {error}
              </div>
            )}
            {success && (
              <div className="bg-green-500/10 border border-green-500/20 text-green-400 text-xs p-3 rounded">
                {success}
              </div>
            )}
          </div>

          {/* Info */}
          <div className="bg-[#1a1a1a] border border-[#2a2a2a] p-4 rounded text-xs text-[#8a8a8a] space-y-2">
            <div className="font-medium text-[#d4d4d4] mb-2">Credit System</div>
            <div>• Each credit corresponds to 1 second of video generated</div>
            <div>• Credits are deducted when videos are generated</div>
            <div>• New users receive 1,000 free trial credits</div>
            <div>• Additional credits cost $0.10 per credit</div>
          </div>
        </div>
      </div>
    </div>
  )
}
