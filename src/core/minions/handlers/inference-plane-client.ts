import type Anthropic from '@anthropic-ai/sdk';

interface OpenAIToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

interface OpenAIChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content?: string | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}

interface OpenAIRequestBody {
  model: string;
  messages: OpenAIChatMessage[];
  max_tokens?: number;
  temperature?: number;
  tools?: Array<{ type: 'function'; function: { name: string; description: string; parameters: Record<string, unknown> } }>;
  tool_choice?: 'auto' | 'none' | { type: 'function'; function: { name: string } };
}

function getBaseUrl(): string {
  return process.env.GBRAIN_INFERENCE_BASE_URL || process.env.LLM_URL || 'http://crsai-vllm:8000/v1';
}

function getApiKey(): string {
  return process.env.GBRAIN_INFERENCE_API_KEY || process.env.OPENAI_API_KEY || 'crsai-internal';
}

function getDefaultModel(): string {
  return process.env.GBRAIN_INFERENCE_MODEL || process.env.LLM_MODEL || 'qwq-32b-q4';
}

function resolveModel(requested: string): string {
  if (requested.startsWith('claude-')) return getDefaultModel();
  return requested;
}

function mapSystem(system: Anthropic.MessageCreateParamsNonStreaming['system']): OpenAIChatMessage[] {
  if (!system) return [];
  const blocks = Array.isArray(system) ? system : [{ type: 'text' as const, text: system }];
  const text = blocks
    .filter((b): b is { type: 'text'; text: string } => typeof b === 'object' && 'type' in b && b.type === 'text')
    .map(b => b.text)
    .join('\n');
  return text ? [{ role: 'system', content: text }] : [];
}

function mapMessages(anthroMessages: Anthropic.MessageParam[]): OpenAIChatMessage[] {
  const out: OpenAIChatMessage[] = [];
  for (const msg of anthroMessages) {
    if (typeof msg.content === 'string') {
      out.push({ role: msg.role as 'user' | 'assistant', content: msg.content });
      continue;
    }
    const text = msg.content.filter(b => b.type === 'text').map(b => (b as any).text).join('\n');
    if (msg.role === 'assistant') {
      const uses = msg.content.filter(b => b.type === 'tool_use') as Array<{ type: 'tool_use'; id: string; name: string; input: unknown }>;
      out.push({ role: 'assistant', content: text || null, tool_calls: uses.map(u => ({ id: u.id, type: 'function', function: { name: u.name, arguments: JSON.stringify(u.input) } })) });
    } else {
      const results = msg.content.filter(b => b.type === 'tool_result') as Array<{ type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean }>;
      if (results.length > 0) {
        for (const r of results) out.push({ role: 'tool', content: r.content, tool_call_id: r.tool_use_id });
      } else if (text) {
        out.push({ role: 'user', content: text });
      }
    }
  }
  return out;
}

function mapTools(tools: Anthropic.Tool[] | undefined): OpenAIRequestBody['tools'] {
  if (!tools || tools.length === 0) return undefined;
  return tools.map(t => ({ type: 'function', function: { name: t.name, description: t.description, parameters: (t as any).input_schema || {} } }));
}

function mapToolChoice(tc: Anthropic.MessageCreateParamsNonStreaming['tool_choice']): OpenAIRequestBody['tool_choice'] {
  if (!tc) return undefined;
  if (tc.type === 'auto') return 'auto';
  if (tc.type === 'none') return 'none';
  if (tc.type === 'tool') return { type: 'function', function: { name: tc.name } };
  return 'auto';
}

function mapResponse(data: any, requestedModel: string): Anthropic.Message {
  const choice = data.choices?.[0];
  if (!choice) throw new Error('No choices in inference-plane response');
  const msg = choice.message;
  const blocks: Anthropic.ContentBlock[] = [];
  if (msg.content) blocks.push({ type: 'text', text: msg.content } as Anthropic.ContentBlock);
  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      let input: unknown;
      try { input = JSON.parse(tc.function.arguments); } catch { input = tc.function.arguments; }
      blocks.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input } as Anthropic.ContentBlock);
    }
  }
  const finish = choice.finish_reason;
  const stopReason: Anthropic.Message['stop_reason'] = finish === 'stop' ? 'end_turn' : finish === 'length' ? 'max_tokens' : finish === 'tool_calls' ? 'tool_use' : null;
  return {
    id: data.id || `infer-${Date.now()}`,
    type: 'message',
    role: 'assistant',
    model: requestedModel,
    stop_reason: stopReason,
    stop_sequence: null,
    content: blocks,
    usage: {
      input_tokens: data.usage?.prompt_tokens ?? 0,
      output_tokens: data.usage?.completion_tokens ?? 0,
      cache_read_input_tokens: 0,
      cache_creation_input_tokens: 0,
    },
  } as Anthropic.Message;
}

/** OpenAI-compatible client that satisfies the Anthropic MessagesClient shape. */
export class InferencePlaneClient {
  async create(params: Anthropic.MessageCreateParamsNonStreaming, opts?: { signal?: AbortSignal }): Promise<Anthropic.Message> {
    const body: OpenAIRequestBody = {
      model: resolveModel(params.model),
      messages: [...mapSystem(params.system), ...mapMessages(params.messages)],
      max_tokens: params.max_tokens,
      temperature: 0.0,
    };
    const tools = mapTools(params.tools as Anthropic.Tool[] | undefined);
    if (tools) body.tools = tools;
    const tc = mapToolChoice(params.tool_choice);
    if (tc) body.tool_choice = tc;
    const res = await fetch(`${getBaseUrl()}/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${getApiKey()}` },
      body: JSON.stringify(body),
      signal: opts?.signal,
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(`Inference plane error ${res.status}: ${text}`);
    }
    return mapResponse(await res.json(), body.model);
  }
}

/** Return Anthropic SDK client or InferencePlaneClient based on GBRAIN_LLM_PROVIDER. */
export function getDefaultMessagesClient(): { create: InferencePlaneClient['create'] } {
  const provider = process.env.GBRAIN_LLM_PROVIDER;
  if (provider === 'inference-plane' || provider === 'openai') {
    return new InferencePlaneClient();
  }
  // Lazy-import Anthropic so the default branch doesn't eagerly load the SDK
  // when the provider is inference-plane.
  const AnthropicSDK = require('@anthropic-ai/sdk').default;
  return new AnthropicSDK().messages;
}
