'use client'
import { useState } from 'react'
import { Card, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

export default function LeadScoring() {
  const [form, setForm] = useState({ engagement: '', time: '', source: 'Social Media', industry: 'Tech' })
  const [prediction, setPrediction] = useState(null)

  const handlePredict = async () => {
    const response = await fetch('http://localhost:8000/predict-lead-score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        engagement_score: parseInt(form.engagement),
        time_spent: parseFloat(form.time),
        source: form.source,
        industry: form.industry
      })
    })
    const data = await response.json()
    setPrediction(data.converted_prediction)
  }

  return (
    <Card>
      <CardContent className="p-6">
        <h1 className="text-2xl font-bold mb-4">Lead Scoring</h1>
        <div className="space-y-4">
          <Input type="number" placeholder="Engagement Score" value={form.engagement} onChange={(e) => setForm({ ...form, engagement: e.target.value })} />
          <Input type="number" placeholder="Time Spent (mins)" value={form.time} onChange={(e) => setForm({ ...form, time: e.target.value })} />
          <Input type="text" placeholder="Source" value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })} />
          <Input type="text" placeholder="Industry" value={form.industry} onChange={(e) => setForm({ ...form, industry: e.target.value })} />
          <Button onClick={handlePredict}>Predict</Button>
          {prediction !== null && <p>Prediction: {prediction ? 'Converted' : 'Not Converted'}</p>}
        </div>
      </CardContent>
    </Card>
  )
}