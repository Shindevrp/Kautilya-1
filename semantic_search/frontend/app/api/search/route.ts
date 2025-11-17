import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { query, top_k = 5, hybrid = true, reranking = true, explain = false } = body

    if (!query || typeof query !== 'string' || query.trim().length === 0) {
      return NextResponse.json(
        { error: 'Query is required and must be a non-empty string' },
        { status: 400 }
      )
    }

    // Get the Python API URL from environment or default to localhost
    const apiUrl = process.env.API_URL || 'http://localhost:8000'

    // Forward the request to the Python API
    const response = await fetch(`${apiUrl}/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: query.trim(),
        top_k: Math.min(Math.max(top_k, 1), 20), // Clamp between 1-20
        hybrid,
        reranking,
        explain,
      }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      return NextResponse.json(
        { error: `API request failed: ${response.status} ${errorText}` },
        { status: response.status }
      )
    }

    const data = await response.json()
    return NextResponse.json(data)

  } catch (error) {
    console.error('Search API error:', error)
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  const query = searchParams.get('query')
  const top_k = parseInt(searchParams.get('top_k') || '5')
  const hybrid = searchParams.get('hybrid') !== 'false'
  const reranking = searchParams.get('reranking') !== 'false'
  const explain = searchParams.get('explain') === 'true'

  // Reuse the POST logic
  const mockRequest = new NextRequest(request.url, {
    method: 'POST',
    body: JSON.stringify({ query, top_k, hybrid, reranking, explain }),
  })

  return POST(mockRequest)
}