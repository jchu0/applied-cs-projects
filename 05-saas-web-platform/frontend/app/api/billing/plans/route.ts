import { NextResponse } from 'next/server';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export async function GET() {
  try {
    const response = await fetch(`${API_BASE_URL}/billing/plans/`, {
      headers: {
        'Content-Type': 'application/json',
      },
      // No auth required for public plans endpoint
    });

    if (!response.ok) {
      // Return empty array if backend unavailable
      return NextResponse.json([]);
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    // Return empty array if backend unavailable (allows fallback to default plans)
    console.error('Failed to fetch plans from backend:', error);
    return NextResponse.json([]);
  }
}
