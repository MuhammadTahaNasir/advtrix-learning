'use client';
import Link from 'next/link';

export default function Home() {
  return (
    <div className="min-h-screen bg-background text-foreground font-sans">
      {/* Header */}
      <header className="flex items-center p-5 bg-sidebar text-sidebar-foreground text-lg font-semibold border-b border-sidebar-border">
        <img
          src="https://img.icons8.com/ios-filled/50/ffffff/instagram-new.png"
          alt="Marketsense Logo"
          className="h-6 mr-2"
        />
        <span>Marketsense Dashboard</span>
      </header>

      {/* Main Content */}
      <main className="text-center px-4 py-16">
        {/* Logo Section */}
        <div className="bg-card p-6 inline-block rounded-full mb-8 border border-border">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-10 w-10 text-primary" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M7 7h10v10H7V7z" />
          </svg>
        </div>

        {/* Title and Subtitle */}
        <h1 className="text-4xl font-bold mb-3">Marketsense Insights</h1>
        <p className="text-muted-foreground text-lg mb-12">
          Explore your data through AI-powered tools, insights, and predictions
        </p>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 px-4 max-w-6xl mx-auto">
          {[
            {
              title: "Churn Analysis",
              href: "/churn-analysis",
              icon: "💬",
              desc: "Understand customer churn trends and risk factors",
            },
            {
              title: "Customer Segmentation",
              href: "/segmentation",
              icon: "👥",
              desc: "Group customers based on behavior and demographics",
            },
            {
              title: "Forecasting",
              href: "/forecasting",
              icon: "📊",
              desc: "Predict future performance using historical data",
            },
            {
              title: "LLM Insights",
              href: "/terrag-insights",
              icon: "🤖",
              desc: "Ask natural language questions about your business",
            },
            {
              title: "Lead Scoring",
              href: "/lead-scoring",
              icon: "⭐",
              desc: "Score and prioritize your sales leads",
            },
          ].map(({ title, href, icon, desc }) => (
            <Link key={title} href={href}>
              <div className="bg-card border border-border p-6 rounded-xl shadow-md hover:translate-y-[-5px] transition-transform cursor-pointer">
                <div className="text-3xl mb-3 text-primary">{icon}</div>
                <h3 className="text-xl font-semibold mb-2">{title}</h3>
                <p className="text-muted-foreground text-sm">{desc}</p>
              </div>
            </Link>
          ))}
        </div>
      </main>
    </div>
  );
}
