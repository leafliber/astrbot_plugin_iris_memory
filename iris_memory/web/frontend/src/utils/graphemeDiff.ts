export type DiffKind = 'equal' | 'insert' | 'delete'

export interface DiffChunk {
  kind: DiffKind
  text: string
  graphemeCount: number
}

const MARK = /^\p{Mark}$/u
const VARIATION_SELECTOR = /[\uFE00-\uFE0F\u{E0100}-\u{E01EF}]/u
const EMOJI_MODIFIER = /[\u{1F3FB}-\u{1F3FF}]/u
const REGIONAL_INDICATOR = /[\u{1F1E6}-\u{1F1FF}]/u
const ZERO_WIDTH_JOINER = '\u200D'

/**
 * Intl.Segmenter 不可用时的保守 fallback。
 *
 * 它覆盖人格文本中最常见的 extended grapheme：组合音标、变体选择符、
 * Emoji 肤色、区域旗帜与 ZWJ Emoji。复杂脚本仍优先交给浏览器原生
 * Intl.Segmenter，fallback 的核心目标是不把 Emoji 展示成破碎码点。
 */
export function fallbackSegmentGraphemes(text: string): string[] {
  const codePoints = Array.from(text)
  const result: string[] = []
  let regionalCount = 0

  for (let i = 0; i < codePoints.length; i++) {
    const current = codePoints[i]
    const previous = result[result.length - 1]

    if (!previous) {
      result.push(current)
      regionalCount = REGIONAL_INDICATOR.test(current) ? 1 : 0
      continue
    }

    if (MARK.test(current) || VARIATION_SELECTOR.test(current) || EMOJI_MODIFIER.test(current)) {
      result[result.length - 1] += current
      continue
    }

    if (current === ZERO_WIDTH_JOINER && i + 1 < codePoints.length) {
      result[result.length - 1] += current + codePoints[++i]
      continue
    }

    if (REGIONAL_INDICATOR.test(current)) {
      if (regionalCount % 2 === 1) {
        result[result.length - 1] += current
      } else {
        result.push(current)
      }
      regionalCount++
      continue
    }

    regionalCount = 0
    result.push(current)
  }

  return result
}

export function segmentGraphemes(text: string, forceFallback = false): string[] {
  const Segmenter = (Intl as unknown as {
    Segmenter?: new (locale?: string, options?: { granularity: string }) => {
      segment(input: string): Iterable<{ segment: string }>
    }
  }).Segmenter

  if (!forceFallback && Segmenter) {
    const segmenter = new Segmenter('zh-CN', { granularity: 'grapheme' })
    return Array.from(segmenter.segment(text), item => item.segment)
  }
  return fallbackSegmentGraphemes(text)
}

interface AtomicDiff {
  kind: DiffKind
  value: string
}

function valueAt(map: Map<number, number>, key: number): number {
  return map.get(key) ?? Number.NEGATIVE_INFINITY
}

/** Myers shortest-edit-script，避免长人格文本使用 O(n*m) 的 LCS 表。 */
function myersDiff(before: string[], after: string[]): AtomicDiff[] {
  if (before.length === 0) return after.map(value => ({ kind: 'insert', value }))
  if (after.length === 0) return before.map(value => ({ kind: 'delete', value }))

  const max = before.length + after.length
  const trace: Array<Map<number, number>> = []
  const frontier = new Map<number, number>([[1, 0]])

  for (let distance = 0; distance <= max; distance++) {
    trace.push(new Map(frontier))
    for (let diagonal = -distance; diagonal <= distance; diagonal += 2) {
      const goDown =
        diagonal === -distance ||
        (diagonal !== distance && valueAt(frontier, diagonal - 1) < valueAt(frontier, diagonal + 1))
      let x = goDown ? valueAt(frontier, diagonal + 1) : valueAt(frontier, diagonal - 1) + 1
      if (!Number.isFinite(x)) x = 0
      let y = x - diagonal

      while (x < before.length && y < after.length && before[x] === after[y]) {
        x++
        y++
      }
      frontier.set(diagonal, x)

      if (x >= before.length && y >= after.length) {
        return backtrack(trace, before, after)
      }
    }
  }
  return []
}

function backtrack(trace: Array<Map<number, number>>, before: string[], after: string[]): AtomicDiff[] {
  const result: AtomicDiff[] = []
  let x = before.length
  let y = after.length

  for (let distance = trace.length - 1; distance >= 0; distance--) {
    const frontier = trace[distance]
    const diagonal = x - y
    const goDown =
      diagonal === -distance ||
      (diagonal !== distance && valueAt(frontier, diagonal - 1) < valueAt(frontier, diagonal + 1))
    const previousDiagonal = goDown ? diagonal + 1 : diagonal - 1
    const previousXRaw = frontier.get(previousDiagonal)
    const previousX = previousXRaw === undefined ? 0 : previousXRaw
    const previousY = previousX - previousDiagonal

    while (x > previousX && y > previousY) {
      result.push({ kind: 'equal', value: before[x - 1] })
      x--
      y--
    }

    if (distance === 0) break
    if (x === previousX) {
      result.push({ kind: 'insert', value: after[y - 1] })
      y--
    } else {
      result.push({ kind: 'delete', value: before[x - 1] })
      x--
    }
  }

  return result.reverse()
}

function mergeAtomicDiff(items: AtomicDiff[]): DiffChunk[] {
  const chunks: DiffChunk[] = []
  for (const item of items) {
    const previous = chunks[chunks.length - 1]
    if (previous?.kind === item.kind) {
      previous.text += item.value
      previous.graphemeCount++
    } else {
      chunks.push({ kind: item.kind, text: item.value, graphemeCount: 1 })
    }
  }
  return chunks
}

export function diffGraphemes(beforeText: string, afterText: string, forceFallback = false): DiffChunk[] {
  const before = segmentGraphemes(beforeText || '', forceFallback)
  const after = segmentGraphemes(afterText || '', forceFallback)
  return mergeAtomicDiff(myersDiff(before, after))
}

export function summarizeDiff(chunks: DiffChunk[]): { inserted: number; deleted: number; unchanged: number } {
  return chunks.reduce(
    (summary, chunk) => {
      if (chunk.kind === 'insert') summary.inserted += chunk.graphemeCount
      else if (chunk.kind === 'delete') summary.deleted += chunk.graphemeCount
      else summary.unchanged += chunk.graphemeCount
      return summary
    },
    { inserted: 0, deleted: 0, unchanged: 0 }
  )
}
