import Link from 'next/link'
import { Home, BarChart, Users, DollarSign, MessageSquare } from 'lucide-react'

export function Sidebar() {
  return (
    <aside className="w-64 bg-white shadow-md p-4">
      <h2 className="text-xl font-bold mb-6">MarketSense AI</h2>
      <nav>
        <ul className="space-y-2">
          <li><Link href="/" className="flex items-center p-2 hover:bg-gray-100"><Home className="mr-2" /> Home</Link></li>
          <li><Link href="/lead-scoring" className="flex items-center p-2 hover:bg-gray-100"><DollarSign className="mr-2" /> Lead Scoring</Link></li>
          <li><Link href="/segmentation" className="flex items-center p-2 hover:bg-gray-100"><Users className="mr-2" /> Segmentation</Link></li>
          <li><Link href="/forecasting" className="flex items-center p-2 hover:bg-gray-100"><BarChart className="mr-2" /> Forecasting</Link></li>
          <li><Link href="/churn-analysis" className="flex items-center p-2 hover:bg-gray-100"><BarChart className="mr-2" /> Churn Analysis</Link></li>
          <li><Link href="/terrag-insights" className="flex items-center p-2 hover:bg-gray-100"><MessageSquare className="mr-2" /> Terrag Insights</Link></li>
        </ul>
      </nav>
    </aside>
  )
}