import { NextResponse } from 'next/server'
import fs from 'fs'
import path from 'path'
import Papa from 'papaparse'

export async function GET() {
    const filePath = path.join(process.cwd(), '..', 'notebooks', 'data', 'churn_analysis_data.csv')
    const fileContent = fs.readFileSync(filePath, 'utf8')
    const parsed = Papa.parse(fileContent, { header: true })
    return NextResponse.json(parsed.data)
}