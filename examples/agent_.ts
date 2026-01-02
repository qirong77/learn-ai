// 1. 导入依赖模块
import { tool } from "@langchain/core/tools";
import { HumanMessage, AIMessage, SystemMessage } from "@langchain/core/messages";
import { MemorySaver, StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { ChatOpenAI } from "@langchain/openai";
import { z } from "zod";
import { LLM_API } from "../API_KEYS";

// 2. 初始化 KIMI LLM（使用 OpenAI 兼容的接口）
const llm = new ChatOpenAI({
  model: "moonshot-v1-8k",
  apiKey: LLM_API.KIMI_API_KEY,
  configuration: {
    baseURL: LLM_API.KIMI_API_BASE,
  },
  temperature: 0.7,
});

// 3. 定义系统提示（Agent角色与行为规则）
const systemPrompt = 
`你是一个专业的天气预报员，喜欢使用双关语。

你可以使用两个工具：
- get_weather_for_location: 用于获取特定位置的天气
- get_user_location: 用于获取用户的位置

如果用户询问天气，确保你知道位置。
如果从问题中可以看出他们指的是他们所在的位置，使用 get_user_location 工具来查找他们的位置。`;

// 4. 定义工具（Agent可调用的函数）
// 4.1 天气查询工具（需传入城市名）
const getWeather = tool(
  async ({ city }) => {
    if(Math.random() > 0.5) {
        return `${city}看起来要下雨了！`;
    }
    return `${city}总是阳光明媚！`;
  },
  {
    name: "get_weather_for_location",
    description: "获取指定城市的天气",
    schema: z.object({
      city: z.string().describe("要查询天气的城市"),
    }),
  }
);

// 4.2 获取用户位置工具
const getUserLocation = tool(
  async () => {
    // 模拟获取用户位置
    return "北京";
  },
  {
    name: "get_user_location",
    description: "获取用户当前所在的位置",
    schema: z.object({}),
  }
);

// 5. 绑定工具到 LLM
const tools = [getWeather, getUserLocation];
const llmWithTools = llm.bindTools(tools);

// 6. 定义 Agent 节点
// 6.1 Agent 节点：调用 LLM
async function callModel(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const systemMessage = new SystemMessage(systemPrompt);
  const response = await llmWithTools.invoke([systemMessage, ...messages]);
  return { messages: [response] };
}

// 6.2 Tool 节点：使用 ToolNode 执行工具调用
const toolNode = new ToolNode(tools);

// 7. 定义路由函数：决定是继续调用工具还是结束
function shouldContinue(state: typeof MessagesAnnotation.State) {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1] as AIMessage;
  
  // 如果有工具调用，继续执行工具
  if (lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "tools";
  }
  // 否则结束
  return END;
}

// 8. 构建状态图（工作流）
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", toolNode)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", shouldContinue, {
    tools: "tools",
    [END]: END,
  })
  .addEdge("tools", "agent");

// 9. 编译图并添加记忆
const memory = new MemorySaver();
const app = workflow.compile({ checkpointer: memory });

// 10. 主函数：运行 Agent
async function main() {
  console.log("🤖 KIMI LLM Agent 启动中...\n");
  
  // 示例对话
  const config = { configurable: { thread_id: "conversation-1" } };
  
  // 第一轮对话
  console.log("用户：北京的天气怎么样？\n");
  let result = await app.invoke(
    { messages: [new HumanMessage("北京的天气怎么样？")] },
    config
  );
  
  const lastMessage = result.messages[result.messages.length - 1];
  console.log(`Agent: ${lastMessage.content}\n`);
  
  // 第二轮对话（测试记忆功能）
  console.log("用户：上海呢？\n");
  result = await app.invoke(
    { messages: [new HumanMessage("上海呢？")] },
    config
  );
  
  const lastMessage2 = result.messages[result.messages.length - 1];
  console.log(`Agent: ${lastMessage2.content}\n`);
  
  console.log("✅ 对话完成！");
}

// 11. 运行
main().catch(console.error);
