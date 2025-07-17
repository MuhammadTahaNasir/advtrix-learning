'use client'
import { Sidebar } from '@/components/Sidebar'
import { useEffect } from 'react'
import { stripUnwantedAttributes } from '@/utils/stripAttributes'

export default function Layout({ children }: { children: React.ReactNode }) {
  useEffect(() => {
    stripUnwantedAttributes();
  }, []);

  return (
    <html lang="en">
      <body className="flex min-h-screen bg-gray-50">
        <Sidebar />
        <main className="flex-1 p-6">{children}</main>
      </body>
    </html>
  )
}