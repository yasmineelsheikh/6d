import { NextRequest, NextResponse } from 'next/server'

// ARES initialization endpoint removed - data loads lazily when needed
export async function POST(request: NextRequest) {
  return NextResponse.json(
    { error: 'ARES initialization endpoint has been removed. Data loads automatically when needed.' },
    { status: 410 } // 410 Gone
  )
}

