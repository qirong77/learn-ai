// 简化版本的天气Agent测试
import { LLM_API } from "./API_KEYS";

// 简单的工具实现
const getUserLocation = () => "Florida";
const getWeatherForLocation = (city: string) => `It's always sunny in ${city}!`;

async function callKimi(prompt: string) {
  const response = await fetch(LLM_API.KIMI_API_BASE + '/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${LLM_API.KIMI_API_KEY}`,
    },
    body: JSON.stringify({
      model: 'moonshot-v1-8k',
      messages: [
        { 
          role: 'system', 
          content: `You are a weather forecaster. When user asks for weather but doesn't specify location, ask for their location. When given a location, provide weather info with puns.` 
        },
        { role: 'user', content: prompt }
      ],
      temperature: 0.7,
      max_tokens: 500
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}

async function simpleWeatherAgent() {
  console.log("=== 简单的天气Agent测试 ===");
  
  // 测试1：询问天气
  const question = "what is the weather outside?";
  console.log(`用户: ${question}`);
  
  let response = await callKimi(question);
  console.log(`Kimi: ${response}`);
  
  // 测试2：如果需要位置，提供位置
  if (response.includes('location') || response.includes('where')) {
    const userLocation = getUserLocation();
    console.log(`\n用户位置: ${userLocation}`);
    
    const weatherPrompt = `The user's location is ${userLocation}. Please provide weather forecast for ${userLocation} with puns.`;
    response = await callKimi(weatherPrompt);
    console.log(`Kimi: ${response}`);
  }
  
  // 测试3：感谢
  console.log(`\n用户: thank you!`);
  response = await callKimi("thank you!");
  console.log(`Kimi: ${response}`);
}

simpleWeatherAgent().catch(console.error);