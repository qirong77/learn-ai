/* 
  RAG Agent 示例
  
  什么是 RAG？
  - RAG（Retrieval-Augmented Generation）检索增强生成
  - 结合了信息检索和语言生成的技术
  - 可以让 LLM 基于外部知识库来回答问题
  
  本示例的工作流程：
  1. 从网页加载文档内容
  2. 将文档切分成小块（chunks）
  3. 将文档块存储到向量数据库
  4. 创建检索工具，让 Agent 可以查询相关信息
  5. Agent 根据检索到的信息回答用户问题
  
  参考文档：https://docs.langchain.com/oss/javascript/langchain/rag
*/

// 1. 导入依赖模块
import "cheerio"; // 用于解析 HTML
import { tool } from "@langchain/core/tools";
import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { MemorySaver, StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { z } from "zod";
import { LLM_API } from "../API_KEYS";
import type { Document } from "@langchain/core/documents";

// 2. 初始化 LLM
const llm = new ChatOpenAI({
  model: "moonshot-v1-8k",
  apiKey: LLM_API.KIMI_API_KEY,
  configuration: {
    baseURL: LLM_API.KIMI_API_BASE,
  },
  temperature: 0.7,
});

// 3. 简单的相似度计算函数（基于关键词匹配）
function calculateSimilarity(text: string, query: string): number {
  const textLower = text.toLowerCase();
  const queryLower = query.toLowerCase();
  const queryWords = queryLower.split(/\s+/);
  
  let score = 0;
  for (const word of queryWords) {
    if (textLower.includes(word)) {
      score += 1;
    }
  }
  
  return score / queryWords.length;
}

// 4. 文档存储（将在加载后存储所有文档块）
let documentChunks: Document[] = [];

// 5. 从网页加载文档并进行处理
console.log("📚 正在加载文档...");

// 4.1 使用 Cheerio 加载网页内容（选择所有 p 标签）
// 5.1 使用 Cheerio 加载网页内容（选择所有 p 标签）
const pTagSelector = "p";
const cheerioLoader = new CheerioWebBaseLoader(
  "https://lilianweng.github.io/posts/2023-06-23-agent/", // 关于 AI Agent 的博客文章
  {
    selector: pTagSelector
  }
);

const docs = await cheerioLoader.load();
console.log(`✅ 成功加载 ${docs.length} 个文档`);

// 5.2 将文档切分成小块（便于检索和处理）
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000,      // 每块最多 1000 个字符
  chunkOverlap: 200      // 块之间重叠 200 个字符，确保上下文连贯
});
const allSplits = await splitter.splitDocuments(docs);
console.log(`✅ 文档已切分为 ${allSplits.length} 个小块`);

// 6. 将文档块存储到内存中
documentChunks = allSplits;
console.log("✅ 文档已加载到内存中\n");

// 7. 定义检索工具（Agent 可以调用此工具来查询相关信息）
const retrieveSchema = z.object({ 
  query: z.string().describe("要查询的问题或关键词")
});

const retrieve = tool(
  async ({ query }) => {
    console.log(`🔎 检索查询: "${query}"`);
    
    // 使用简单的关键词匹配来查找相关文档
    const scoredDocs = documentChunks.map(doc => ({
      doc,
      score: calculateSimilarity(doc.pageContent, query)
    }));
    
    // 按相似度排序并取前 2 个
    scoredDocs.sort((a, b) => b.score - a.score);
    const retrievedDocs = scoredDocs.slice(0, 2).map(item => item.doc);
    
    // 将检索到的文档格式化为字符串
    const serialized = retrievedDocs
      .map(
        (doc, index) => 
          `[文档片段 ${index + 1}]\n来源: ${doc.metadata.source}\n内容: ${doc.pageContent}\n`
      )
      .join("\n");
    
    console.log(`✅ 找到 ${retrievedDocs.length} 个相关文档片段\n`);
    return serialized;
  },
  {
    name: "retrieve",
    description: "从知识库中检索与查询相关的信息。当需要回答关于 AI Agent 的问题时使用此工具。",
    schema: retrieveSchema,
  }
);

// 8. 定义系统提示
const systemPrompt = 
`你是一个专业的 AI 助手，专门回答关于 AI Agent 的问题。

你可以使用 retrieve 工具来查询知识库中的相关信息。

回答问题时：
1. 首先使用 retrieve 工具查找相关信息
2. 基于检索到的内容来回答问题
3. 如果检索到的信息不足以回答问题，请说明
4. 用清晰、简洁的中文回答`;

// 9. 绑定工具到 LLM
const tools = [retrieve];
const llmWithTools = llm.bindTools(tools);

// 10. 定义 Agent 节点
async function callModel(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const systemMessage = new SystemMessage(systemPrompt);
  const response = await llmWithTools.invoke([systemMessage, ...messages]);
  return { messages: [response] };
}

// 11. 定义工具节点
const toolNode = new ToolNode(tools);

// 12. 定义路由函数
function shouldContinue(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as any;
  
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "tools";
  }
  return END;
}

// 13. 构建状态图
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, {
    tools: "tools",
    [END]: END,
  })
  .addEdge("tools", "agent");

// 14. 编译图
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });

// 15. 主函数
async function main() {
  console.log("🤖 RAG Agent 已启动！\n");
  console.log("=" .repeat(60));
  
  const config = { configurable: { thread_id: "rag-conversation-1" } };
  
  // 示例问题 1
  const question1 = "什么是 AI Agent？它有哪些关键组件？";
  console.log(`\n👤 用户: ${question1}\n`);
  
  let result = await app.invoke(
    { messages: [new HumanMessage(question1)] },
    config
  );
  
  let lastMessage = result.messages[result.messages.length - 1];
  console.log(`🤖 Agent: ${lastMessage.content}\n`);
  console.log("=" .repeat(60));
  
  // 示例问题 2
  const question2 = "Agent 中的记忆（Memory）有什么作用？";
  console.log(`\n👤 用户: ${question2}\n`);
  
  result = await app.invoke(
    { messages: [new HumanMessage(question2)] },
    config
  );
  
  lastMessage = result.messages[result.messages.length - 1];
  console.log(`🤖 Agent: ${lastMessage.content}\n`);
  console.log("=" .repeat(60));
  
  console.log("\n✅ RAG Agent 演示完成！");
  console.log("\n💡 提示: RAG 技术让 AI 能够基于外部知识库回答问题，");
  console.log("   避免了模型训练数据过时的问题，并能提供更准确的答案。");
}

// 16. 运行
main().catch(console.error);

