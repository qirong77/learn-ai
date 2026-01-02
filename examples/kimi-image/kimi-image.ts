import { ChatOpenAI } from "@langchain/openai";
import { LLM_API } from "../../API_KEYS";
import { readFileSync } from "node:fs";
import path from "node:path";
import { HumanMessage } from "@langchain/core/messages";

// 1. 创建支持视觉的聊天模型
const client = new ChatOpenAI({
    apiKey: LLM_API.KIMI_API_KEY,
    configuration: {
        baseURL: LLM_API.KIMI_API_BASE,
    },
    model: 'moonshot-v1-8k-vision-preview', // 使用视觉模型
});

// 2. 核心函数：识别图片内容
async function recognizeImage(imagePath: string) {
  try {
    // 读取图片文件（同步读取）
    const imageData = readFileSync(imagePath);
    
    // 获取图片扩展名（如 png/jpg，去掉前面的点）
    const ext = path.extname(imagePath).slice(1);
    if (!ext) {
      throw new Error('图片文件没有扩展名，请检查文件路径');
    }

    // 3. 将二进制图片数据转为Base64，并拼接成Data URL
    const base64Str = imageData.toString('base64');
    const imageUrl = `data:image/${ext};base64,${base64Str}`;

    // 4. 调用Moonshot视觉API
    // 构建包含图片的消息
    const message = new HumanMessage({
      content: [
        {
          type: 'image_url', // 图片类型
          image_url: {
            url: imageUrl, // Base64格式的图片URL
          },
        },
        {
          type: 'text', // 文字指令类型
          text: '请详细描述图片的内容。', // 识别图片的指令
        },
      ],
    });

    // 调用模型
    const response = await client.invoke([message]);

    // 输出识别结果
    console.log('图片识别结果：\n', response.content);
    return response.content;

  } catch (error) {
    // 错误处理
    console.error('识别图片失败：', (error as Error).message);
    throw error; // 抛出错误供上层处理
  }
}

// 5. 执行函数（使用相对路径）
const imagePath = path.join(__dirname, 'image.png'); // 当前目录下的图片文件
recognizeImage(imagePath);