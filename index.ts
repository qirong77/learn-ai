// 1. 导入依赖模块
import { tool } from "@langchain/core/tools";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { MemorySaver, StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { z } from "zod";
import { LLM_API } from "./API_KEYS";

// 2. 定义系统提示（Agent角色与行为规则）
const systemPrompt = `You are an expert weather forecaster, who speaks in puns.

You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. 
If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

// 3. 定义工具（Agent可调用的函数）
// 3.1 天气查询工具（需传入城市名）
const getWeather = tool(
  async ({ city }) => {
    return `It's always sunny in ${city}!`; // 实际场景可替换为真实天气API调用
  },
  {
    name: "get_weather_for_location",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string().describe("The city to get the weather for"),
    }),
  }
);

// 3.2 用户位置获取工具（基于用户ID返回默认位置，实际场景可对接用户系统）
const getUserLocation = tool(
  async (_input, config) => {
    // 从配置中获取用户ID，默认为"1"
    const user_id = config?.configurable?.user_id || "1";
    // 模拟逻辑：user_id=1返回Florida，其他返回SF
    return user_id === "1" ? "Florida" : "SF";
  },
  {
    name: "get_user_location",
    description: "Retrieve user's location based on user ID",
    schema: z.object({}), // 无输入参数
  }
);

// 4. 创建 KIMI 模型调用函数
async function callKimiModel(messages: any[]) {
  // 转换消息格式
  const kimiMessages = [
    { role: 'system', content: systemPrompt },
    ...messages.map(msg => ({
      role: msg.constructor.name === 'HumanMessage' ? 'user' : 'assistant',
      content: msg.content || ''
    })).filter(msg => msg.content.trim())
  ];

  const response = await fetch(LLM_API.KIMI_API_BASE + '/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${LLM_API.KIMI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'moonshot-v1-8k',
      messages: kimiMessages,
      temperature: 0.7,
      max_tokens: 500
    })
  });

  if (!response.ok) {
    throw new Error(`KIMI API错误: ${response.status}`);
  }

  const data = await response.json();
  return new AIMessage({ content: data.choices[0].message.content });
}

// 5. 工具调用函数
const tools = [getUserLocation, getWeather];

// 6. 配置对话记忆（维持多轮交互状态）
const checkpointer = new MemorySaver(); // 开发环境用；生产环境建议替换为数据库持久化存储

// 7. 创建状态图（Agent驱动的交互流程）
const workflow = new StateGraph(MessagesAnnotation)
  .addNode("agent", async (state) => {
    const response = await callKimiModel(state.messages);
    return { messages: [response] };
  })
  .addNode("tools", async (state) => {
    const lastMessage = state.messages[state.messages.length - 1] as any;
    const toolCalls = lastMessage.tool_calls || [];
    const toolResults = [];
    
    for (const toolCall of toolCalls) {
      const tool = tools.find(t => t.name === toolCall.name);
      if (tool) {
        const result = tool.name === "get_weather_for_location"
          ? await getWeather.func({ city: toolCall.args.city })
          : await getUserLocation.func({});
        
        toolResults.push({
          type: "tool" as const,
          name: toolCall.name,
          tool_call_id: toolCall.id,
          content: String(result)
        });
      }
    }
    
    return { messages: toolResults };
  })
  .addEdge(START, "agent")
  .addConditionalEdges(
    "agent",
    (state) => {
      const lastMessage = state.messages[state.messages.length - 1] as any;
      return lastMessage.tool_calls && lastMessage.tool_calls.length > 0 ? "tools" : END;
    },
    {
      tools: "tools",
      [END]: END
    }
  )
  .addEdge("tools", "agent");

// 8. 编译Agent
const agent = workflow.compile({ checkpointer });

// 9. 运行Agent（多轮对话示例）
async function runWeatherAgent() {
  // 对话配置：唯一thread_id（用于维持会话状态）、用户ID
  const config = {
    configurable: { 
      thread_id: "unique-conversation-id-123",
      user_id: "1" // 用户ID用于获取位置
    },
  };

  // 第一轮：询问天气
  console.log("用户: what is the weather outside?");
  const firstResponse = await agent.invoke(
    { messages: [new HumanMessage("what is the weather outside?")] }, 
    config
  );
  console.log("助手:", firstResponse.messages[firstResponse.messages.length - 1].content);

  // 第二轮：感谢
  console.log("\n用户: thank you!");
  const secondResponse = await agent.invoke(
    { messages: [new HumanMessage("thank you!")] }, 
    config
  );
  console.log("助手:", secondResponse.messages[secondResponse.messages.length - 1].content);
}

// 执行Agent
runWeatherAgent().catch((error) => {
  console.error("Agent运行出错：", error);
});