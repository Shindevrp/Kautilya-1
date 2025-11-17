'use client'

import { ChangeEvent, FormEvent, useState } from 'react'
import {
  Loader2,
  FileText,
  Route,
  Share2,
  RefreshCw,
  AlertTriangle,
} from 'lucide-react'

type TimelineItem = {
  date: string | null
  headline: string
  url: string
  score: number
  why_it_matters: string
}

type ClusterMember = {
  index: number
  score: number
  headline: string
  url: string
  snippet: string
}

type Cluster = {
  cluster_id: number
  size: number
  representative: ClusterMember
  members: ClusterMember[]
}

type GraphNode = {
  id: number
  headline: string
  url: string
  date: string | null
}

type GraphEdge = {
  source: number
  target: number
  score: number
  relation: string
}

type NarrativeResponse = {
  narrative_summary: string
  timeline: TimelineItem[]
  clusters: Cluster[]
  graph: {
    nodes: GraphNode[]
    edges: GraphEdge[]
  }
  metadata: {
    topic: string
    source_count: number
    selected_count: number
  }
}

const DEFAULT_TOPIC = 'AI regulation'
const DEFAULT_JSON_PATH = 'news.json'

export default function NarrativeDashboard() {
  const [topic, setTopic] = useState(DEFAULT_TOPIC)
  const [jsonPath, setJsonPath] = useState(DEFAULT_JSON_PATH)
  const [jsonContent, setJsonContent] = useState('')
  const [minRating, setMinRating] = useState(8)
  const [topK, setTopK] = useState(200)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<NarrativeResponse | null>(null)

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    if (!topic.trim()) {
      setError('Topic is required')
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const payload: Record<string, unknown> = {
        topic: topic.trim(),
        minRating,
        topK,
      }

      if (jsonContent.trim().length > 0) {
        payload.jsonContent = jsonContent
      } else {
        payload.jsonPath = jsonPath.trim() || DEFAULT_JSON_PATH
      }

      const response = await fetch('/api/narrative', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || 'Failed to build narrative')
        setResult(null)
        return
      }

      setResult(data as NarrativeResponse)
    } catch (fetchError) {
      setError(fetchError instanceof Error ? fetchError.message : 'Unknown error')
      setResult(null)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 py-9">
          <h1 className="text-3xl font-bold text-slate-900 tracking-tight">
            Narrative Builder Explorer
          </h1>
          <p className="mt-2 text-slate-600 max-w-3xl">
            Run the Python narrative builder directly from the browser. Provide a topic plus a
            dataset path (or paste JSON inline) to generate timelines, clusters, and a narrative
            graph that you can explore visually.
          </p>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-6 py-10 space-y-10">
        <section className="card p-8">
          <form className="grid gap-6" onSubmit={handleSubmit}>
            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">Topic</label>
              <input
                value={topic}
                onChange={(event: ChangeEvent<HTMLInputElement>) => setTopic(event.target.value)}
                placeholder="AI regulation"
              />
            </div>

            <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
              <div>
                <label className="block text-sm font-semibold text-slate-700 mb-2">
                  JSON dataset path
                </label>
                <input
                  value={jsonPath}
                  onChange={(event: ChangeEvent<HTMLInputElement>) => setJsonPath(event.target.value)}
                  placeholder="news.json"
                  disabled={jsonContent.trim().length > 0}
                />
                <p className="mt-2 text-xs text-slate-500">
                  Path is resolved relative to the Narrative_Building directory. Leave blank to use the
                  bundled <code>news.json</code> sample. Paste JSON below to override the path.
                </p>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <label className="block text-sm font-semibold text-slate-700">
                  <span className="mb-2 block">Minimum rating</span>
                  <input
                    type="number"
                    step="0.1"
                    value={minRating}
                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                      setMinRating(Number(event.target.value))
                    }
                    min={0}
                    max={10}
                  />
                </label>

                <label className="block text-sm font-semibold text-slate-700">
                  <span className="mb-2 block">Top K</span>
                  <input
                    type="number"
                    value={topK}
                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                      setTopK(Number(event.target.value))
                    }
                    min={10}
                    max={2000}
                  />
                </label>
              </div>
            </div>

            <div>
              <label className="block text-sm font-semibold text-slate-700 mb-2">Inline JSON</label>
              <textarea
                rows={6}
                value={jsonContent}
                onChange={(event: ChangeEvent<HTMLTextAreaElement>) =>
                  setJsonContent(event.target.value)
                }
                placeholder="Paste JSON or NDJSON here to run without referencing a file on disk"
              />
              <p className="mt-2 text-xs text-slate-500">
                When this field has content it takes precedence over the dataset path. Files are saved
                temporarily and removed after the run.
              </p>
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-xs text-slate-500">
                <Share2 className="h-4 w-4" />
                <span>Python entry point: <code>narrative_builder.py</code></span>
              </div>

              <button
                type="submit"
                className="btn-primary flex items-center gap-2"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Building narrative…
                  </>
                ) : (
                  <>
                    <RefreshCw className="h-4 w-4" />
                    Run narrative builder
                  </>
                )}
              </button>
            </div>

            {error && (
              <div className="border border-amber-200 bg-amber-50 text-amber-800 rounded-lg px-4 py-3 flex items-start gap-3">
                <AlertTriangle className="h-5 w-5 mt-0.5" />
                <div>
                  <p className="font-semibold">Run failed</p>
                  <p className="text-sm leading-tight">{error}</p>
                </div>
              </div>
            )}
          </form>
        </section>

        {result && (
          <section className="space-y-8">
            <article className="card p-8 space-y-4">
              <header className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary-600" />
                <h2 className="text-xl font-semibold text-slate-900">Narrative summary</h2>
              </header>
              <p className="text-slate-700 leading-relaxed">{result.narrative_summary}</p>
              <div className="text-xs text-slate-500 flex flex-wrap gap-4">
                <span>Topic: {result.metadata.topic}</span>
                <span>Sources scanned: {result.metadata.source_count}</span>
                <span>Selected: {result.metadata.selected_count}</span>
              </div>
            </article>

            <article className="card p-8">
              <header className="flex items-center gap-2 mb-6">
                <Route className="h-5 w-5 text-primary-600" />
                <h2 className="text-xl font-semibold text-slate-900">Chronological timeline</h2>
              </header>

              {result.timeline.length === 0 ? (
                <p className="text-sm text-slate-500">No events surfaced for this topic.</p>
              ) : (
                <div className="space-y-6">
                  {result.timeline.map((item, index) => (
                    <div key={`${item.headline}-${index}`} className="border-l-2 border-primary-200 pl-4">
                      <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500 mb-1">
                        <span className="font-semibold text-primary-700">
                          {item.date ? new Date(item.date).toLocaleDateString() : 'Undated'}
                        </span>
                        <span>Score: {item.score.toFixed(3)}</span>
                        {item.url && (
                          <a
                            href={item.url}
                            target="_blank"
                            rel="noreferrer"
                            className="text-primary-600 underline"
                          >
                            Source link
                          </a>
                        )}
                      </div>
                      <p className="font-medium text-slate-800">{item.headline || 'Untitled item'}</p>
                      <p className="text-sm text-slate-600 mt-1 leading-relaxed">
                        {item.why_it_matters}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </article>

            <article className="card p-8">
              <header className="flex items-center gap-2 mb-6">
                <Share2 className="h-5 w-5 text-primary-600" />
                <h2 className="text-xl font-semibold text-slate-900">Narrative clusters</h2>
              </header>

              {result.clusters.length === 0 ? (
                <p className="text-sm text-slate-500">Clustering did not yield any meaningful groups.</p>
              ) : (
                <div className="grid gap-4 md:grid-cols-2">
                  {result.clusters.map((cluster) => (
                    <div key={cluster.cluster_id} className="border border-slate-200 rounded-lg p-4">
                      <div className="flex items-baseline justify-between">
                        <span className="text-sm font-semibold text-primary-700">
                          Cluster #{cluster.cluster_id}
                        </span>
                        <span className="text-xs text-slate-500">{cluster.size} article(s)</span>
                      </div>
                      <p className="mt-2 text-sm text-slate-600 leading-relaxed">
                        {cluster.representative.snippet || cluster.representative.headline}
                      </p>
                      <ul className="mt-3 space-y-1 text-xs text-slate-500">
                        {cluster.members.slice(0, 5).map((member) => (
                          <li key={`${cluster.cluster_id}-${member.index}`}>
                            <span className="font-medium text-slate-600">{member.score.toFixed(3)}</span>
                            {' – '}
                            {member.headline || 'Untitled'}
                          </li>
                        ))}
                        {cluster.members.length > 5 && (
                          <li className="italic">… {cluster.members.length - 5} more</li>
                        )}
                      </ul>
                    </div>
                  ))}
                </div>
              )}
            </article>

            <article className="card p-8">
              <header className="flex items-center gap-2 mb-6">
                <Share2 className="h-5 w-5 text-primary-600" />
                <h2 className="text-xl font-semibold text-slate-900">Graph connections</h2>
              </header>

              {result.graph.nodes.length === 0 ? (
                <p className="text-sm text-slate-500">Graph could not be constructed for the selected items.</p>
              ) : (
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700 mb-2">Nodes</h3>
                    <ul className="space-y-2 text-sm text-slate-600">
                      {result.graph.nodes.map((node) => (
                        <li key={node.id} className="border border-slate-200 rounded-md p-3">
                          <p className="font-medium text-slate-700 mb-1">
                            #{node.id} {node.headline || 'Untitled'}
                          </p>
                          <div className="text-xs text-slate-500 flex flex-wrap gap-3">
                            <span>{node.date ? new Date(node.date).toLocaleDateString() : 'Undated'}</span>
                            {node.url && (
                              <a
                                href={node.url}
                                target="_blank"
                                rel="noreferrer"
                                className="text-primary-600 underline"
                              >
                                Link
                              </a>
                            )}
                          </div>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-slate-700 mb-2">Edges</h3>
                    <ul className="space-y-2 text-sm text-slate-600">
                      {result.graph.edges.map((edge, index) => (
                        <li key={`${edge.source}-${edge.target}-${index}`} className="border border-slate-200 rounded-md p-3">
                          <p className="font-medium text-slate-700">
                            {edge.source} → {edge.target}
                          </p>
                          <p className="text-xs text-slate-500">
                            Relation: {edge.relation} · Score: {edge.score.toFixed(3)}
                          </p>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
            </article>
          </section>
        )}
      </main>
    </div>
  )
}
