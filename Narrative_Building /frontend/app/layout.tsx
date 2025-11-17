import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Narrative Builder Dashboard',
  description: 'Interactive UI for exploring narrative summaries from large news datasets.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  )
}
