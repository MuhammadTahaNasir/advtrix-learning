'use client'
import { useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Sparkles } from 'lucide-react'

export default function TerragInsights() {
    const [query, setQuery] = useState('')
    const [answer, setAnswer] = useState(null)

    const handleQuery = async () => {
        const response = await fetch('http://localhost:8000/ask-insight', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        })
        const data = await response.json()
        setAnswer(data.answer)
    }

    return (
        <Card>
            <CardContent className="p-6">
                <h1 className="text-2xl font-bold mb-4">Terrag Insights</h1>
                <div className="space-y-4">
                    <Input placeholder="Ask a marketing question..." value={query} onChange={(e) => setQuery(e.target.value)} />
                    <Button onClick={handleQuery}><Sparkles className="mr-2" /> Get Insights</Button>
                    {answer && <p>{answer}</p>}
                </div>
            </CardContent>
        </Card>
    )
}