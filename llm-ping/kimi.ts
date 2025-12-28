// 测试KIMI API的简单示例
import { LLM_API } from "../API_KEYS";

async function testKimiAPI() {
  try {
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
            role: 'user',
            content: '你好，请介绍一下你是谁'
          }
        ],
        temperature: 0.3
      })
    });

    if (!response.ok) {
      const error = await response.text();
      console.error('API错误:', error);
      return;
    }

    const data = await response.json();
    console.log('KIMI API测试成功:');
    console.log(data.choices[0].message.content);
  } catch (error) {
    console.error('请求失败:', error);
  }
}

testKimiAPI();