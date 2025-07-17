'use client'
import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent } from '@/components/ui/card'
import { SegmentationData } from '@/types/segmentationData'
import type { PlotData, Layout } from 'plotly.js'

interface PlotProps {
  data: Partial<PlotData>[]
  layout: Partial<Layout>
  style?: React.CSSProperties
}

const Plot = dynamic<PlotProps>(() => import('react-plotly.js').then(mod => mod.default), { ssr: false })

export default function Segmentation() {
    const [data, setData] = useState<SegmentationData[]>([])

    useEffect(() => {
        fetch('/api/segmentation')
            .then(res => res.json())
            .then(data => setData(data))
            .catch(err => console.error('Failed to fetch segmentation data:', err))
    }, [])

    if (data.length === 0) return <div>Loading...</div>

    const plotData: Partial<PlotData>[] = [
        {
            x: data.map(d => parseFloat(d.age as string)),
            y: data.map(d => parseFloat(d.income as string)),
            mode: 'markers',
            type: 'scatter',
            marker: {
                color: data.map(d => ['#FF6B6B', '#4ECDC4', '#45B7D1'][parseInt(d.cluster as string)]),
                line: { color: '#262730', width: 1 }
            }
        }
    ]

    const layout: Partial<Layout> = {
        title: { text: 'Customer Clusters by Age and Income' },
        xaxis: { title: { text: 'Age' } },
        yaxis: { title: { text: 'Income' } }
    }

    return (
        <Card>
            <CardContent className="p-6">
                <h1 className="text-2xl font-bold mb-4">Customer Segmentation</h1>
                <Plot
                    data={plotData}
                    layout={layout}
                    style={{ width: '100%', height: '400px' }}
                />
            </CardContent>
        </Card>
    )
}