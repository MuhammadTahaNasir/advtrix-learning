'use client'
import { useState, useEffect } from 'react'
import dynamic from 'next/dynamic'
import { Card, CardContent } from '@/components/ui/card'
import { ForecastData } from '@/types/forecastData'
import type { PlotData, Layout } from 'plotly.js'

interface PlotProps {
  data: Partial<PlotData>[]
  layout: Partial<Layout>
  style?: React.CSSProperties
}

const Plot = dynamic<PlotProps>(() => import('react-plotly.js').then(mod => mod.default), { ssr: false })

export default function Forecasting() {
    const [data, setData] = useState<ForecastData[]>([])

    useEffect(() => {
        fetch('/api/forecasting')
            .then(res => res.json())
            .then(data => setData(data))
            .catch(err => console.error('Failed to fetch forecast data:', err))
    }, [])

    if (data.length === 0) return <div>Loading...</div>

    const plotData: Partial<PlotData>[] = [
        {
            x: data.map(d => d.ds),
            y: data.map(d => parseFloat(d.yhat as string)),
            type: 'scatter',
            mode: 'lines',
            name: 'Forecast',
            line: { color: '#FF6B6B' }
        },
        {
            x: data.map(d => d.ds),
            y: data.map(d => parseFloat(d.yhat_lower as string)),
            type: 'scatter',
            mode: 'lines',
            name: 'Confidence Interval Lower',
            line: { color: 'rgba(68, 68, 68, 0.3)', dash: 'dash' }
        },
        {
            x: data.map(d => d.ds),
            y: data.map(d => parseFloat(d.yhat_upper as string)),
            type: 'scatter',
            mode: 'lines',
            name: 'Confidence Interval Upper',
            line: { color: 'rgba(68, 68, 68, 0.3)', dash: 'dash' },
            fill: 'tonexty',
            fillcolor: 'rgba(68, 68, 68, 0.3)'
        }
    ]

    const layout: Partial<Layout> = {
        title: { text: 'Conversions Forecast' },
        xaxis: { title: { text: 'Date' } },
        yaxis: { title: { text: 'Conversions' } }
    }

    return (
        <Card>
            <CardContent className="p-6">
                <h1 className="text-2xl font-bold mb-4">Marketing Trends Forecast</h1>
                <Plot
                    data={plotData}
                    layout={layout}
                    style={{ width: '100%', height: '400px' }}
                />
            </CardContent>
        </Card>
    )
}