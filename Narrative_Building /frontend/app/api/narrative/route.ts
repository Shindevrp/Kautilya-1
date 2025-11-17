import { NextResponse } from 'next/server'
import { spawn } from 'child_process'
import path from 'path'
import { promises as fs } from 'fs'

interface NarrativeRequestBody {
  topic?: string
  jsonPath?: string
  jsonContent?: string
  minRating?: number
  topK?: number
}

const DEFAULT_JSON = 'news.json'
const DEFAULT_MIN_RATING = 8
const DEFAULT_TOPK = 300

async function runPython(pythonCmd: string, args: string[], cwd: string) {
  return new Promise<string>((resolve, reject) => {
    const child = spawn(pythonCmd, args, {
      cwd,
      env: process.env,
    })

    let stdout = ''
    let stderr = ''

    child.stdout.on('data', (data: Buffer) => {
      stdout += data.toString()
    })

    child.stderr.on('data', (data: Buffer) => {
      stderr += data.toString()
    })

    child.on('error', (error: Error) => {
      reject(error)
    })

    child.on('close', (code: number | null) => {
      if (code !== 0) {
        reject(new Error(stderr || `Python process exited with code ${code}`))
        return
      }
      resolve(stdout)
    })
  })
}

export async function POST(req: Request) {
  const body = (await req.json()) as NarrativeRequestBody
  const topic = body.topic?.trim()
  const minRating = body.minRating ?? DEFAULT_MIN_RATING
  const topK = body.topK ?? DEFAULT_TOPK
  const jsonPath = body.jsonPath?.trim() || DEFAULT_JSON
  const jsonContent = body.jsonContent

  if (!topic) {
    return NextResponse.json({ error: 'Topic is required.' }, { status: 400 })
  }

  const projectRoot = path.join(process.cwd(), '..')
  const scriptPath = path.join(projectRoot, 'narrative_builder.py')
  let datasetPath = ''
  let tempFilePath: string | null = null

  try {
    if (jsonContent && jsonContent.trim().length > 0) {
      const tempPath = path.join(projectRoot, '.tmp_narrative_input.json')
      await fs.writeFile(tempPath, jsonContent, 'utf-8')
      tempFilePath = tempPath
      datasetPath = tempPath
    } else {
      datasetPath = path.isAbsolute(jsonPath)
        ? jsonPath
        : path.join(projectRoot, jsonPath)
    }

    await fs.access(datasetPath)

    const pythonCmd = process.env.NARRATIVE_PYTHON || 'python3'
    const args = [
      scriptPath,
      '--topic',
      topic,
      '--json',
      datasetPath,
      '--min_rating',
      String(minRating),
      '--topk',
      String(topK),
    ]

    const rawOutput = await runPython(pythonCmd, args, projectRoot)

    try {
      const parsed = JSON.parse(rawOutput)
      return NextResponse.json(parsed)
    } catch (parseError) {
      return NextResponse.json(
        {
          error: 'Failed to parse narrative_builder output. See server logs for details.',
          rawOutput,
        },
        { status: 500 },
      )
    }
  } catch (error: unknown) {
    const message =
      error instanceof Error ? error.message : 'Unexpected error while running narrative_builder.'
    return NextResponse.json({ error: message }, { status: 500 })
  } finally {
    if (tempFilePath) {
      try {
        await fs.unlink(tempFilePath)
      } catch (cleanupError) {
        console.warn('Failed to clean up temporary narrative input file:', cleanupError)
      }
    }
  }
}
