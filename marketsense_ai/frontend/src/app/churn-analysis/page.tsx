'use client'
import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent } from '@/components/ui/card'
import { ChurnData } from '@/types/churnData'
import type { PlotData, Layout } from 'plotly.js'

interface PlotProps {
  data: Partial<PlotData>[]
  layout: Partial<Layout>
  style?: React.CSSProperties
}

const Plot = dynamic<PlotProps>(() => import('react-plotly.js').then(mod => mod.default), { ssr: false })

export default function ChurnAnalysis() {
    const [data, setData] = useState<ChurnData[]>([])

    useEffect(() => {
        fetch('/api/churn')
            .then(res => res.json())
            .then(data => setData(data))
            .catch(err => console.error('Failed to fetch churn data:', err))
    }, [])

    if (data.length === 0) return <div>Loading...</div>

    const bins = [0, 12, 24, 36, 48, 60]
    const labels = ['0-12', '12-24', '24-36', '36-48', '48-60']

    const binnedData = data.map(d => {
        const tenure = parseFloat(d.tenure_months as string)
        for (let i = 0; i < bins.length - 1; i++) {
            if (tenure >= bins[i] && tenure < bins[i + 1]) {
                return { ...d, tenure_bin: labels[i] }
            }
        }
        return { ...d, tenure_bin: '60+' }
    })

    const churnRates: number[] = labels.map(label => {
        const binData = binnedData.filter(d => d.tenure_bin === label)
        const churned = binData.filter(d => parseInt(d.churned as string) === 1).length
        const total = binData.length
        return total > 0 ? Number((churned / total * 100).toFixed(2)) : 0
    })

    const plotData: Partial<PlotData>[] = [
        {
            x: labels,
            y: churnRates,
            type: 'bar',
            marker: {
                color: '#FF6B6B',
                line: { color: '#262730', width: 1 }
            },
            name: 'Churn Rate (%)'
        }
    ]

    const layout: Partial<Layout> = {
        title: { text: 'Churn Rate by Tenure Bin' },
        xaxis: { title: { text: 'Tenure Bin (Months)' } },
        yaxis: { title: { text: 'Churn Rate (%)' } }
    }

    return (
        <Card>
            <CardContent className="p-6">
                <h1 className="text-2xl font-bold mb-4">Churn Analysis</h1>
                <Plot
                    data={plotData}
                    layout={layout}
                    style={{ width: '100%', height: '400px' }}
                />
            </CardContent>
        </Card>
    )
}