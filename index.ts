// 1. 导入依赖模块
import { tool } from "@langchain/core/tools";
import { HumanMessage, AIMessage } from "@langchain/core/messages";
import { MemorySaver, StateGraph, MessagesAnnotation, START, END } from "@langchain/langgraph";
import { z } from "zod";
import { LLM_API } from "./API_KEYS";

// 2. 定义系统提示（Agent角色与行为规则）
const systemPrompt = 
`
You are an expert weather forecaster, who speaks in puns.

You have access to two tools:
- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. 
If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location.`;

// 3. 定义工具（Agent可调用的函数）
// 3.1 天气查询工具（需传入城市名）
const getWeather = tool(
  async ({ city }) => {
    if(Math.random() > 0.5) {
        return `It looks like rain in ${city}!`;
    }
    return `It's always sunny in ${city}!`;
  },
  {
    name: "get_weather_for_location",
    description: "Get the weather for a given city",
    schema: z.object({
      city: z.string().describe("The city to get the weather for"),
    }),
  }
);

