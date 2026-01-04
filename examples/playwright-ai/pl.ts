import { chromium } from "playwright";

const browser = await chromium.launch({ headless: false });
const page = await browser.newPage();
await page.goto("file:///Users/qironglin/Desktop/playwright/index.html");

// 等待页面加载完成
await page.waitForLoadState('domcontentloaded');

// 获取 canvas 的位置
const canvasBox = await page.locator('#myCanvas').boundingBox();
if (canvasBox) {
    console.log('Canvas 位置信息:');
    console.log(`  x: ${canvasBox.x}`);
    console.log(`  y: ${canvasBox.y}`);
    console.log(`  width: ${canvasBox.width}`);
    console.log(`  height: ${canvasBox.height}`);
    console.log(`  中心点: (${canvasBox.x + canvasBox.width / 2}, ${canvasBox.y + canvasBox.height / 2})`);
}

// 等待用户点击"开始作画"按钮
await page.locator('#drawButton').click();

// 在 canvas 中心画一个圆圈
if (canvasBox) {
    const centerX = canvasBox.x + canvasBox.width / 2;
    const centerY = canvasBox.y + canvasBox.height / 2;
    const radius = 50; // 圆的半径
    
    console.log('\n开始在 canvas 中心画圆...');
    
    // 首先点击 canvas 使其聚焦
    await page.mouse.click(centerX, centerY);
    await page.waitForTimeout(500);
    
    // 画圆：从圆的最右边开始，逆时针画一圈
    const steps = 50; // 圆圈的平滑度
    
    // 移动到起始点（圆的最右边）
    await page.mouse.move(centerX + radius, centerY);
    await page.mouse.down();
    
    // 逐步画圆
    for (let i = 0; i <= steps; i++) {
        const angle = (i / steps) * 2 * Math.PI;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        await page.mouse.move(x, y);
        await page.waitForTimeout(20); // 控制绘制速度
    }
    
    await page.mouse.up();
    console.log('圆圈绘制完成！');
}

// 保持浏览器打开，方便查看结果
console.log('\n浏览器将保持打开状态，可以继续手动绘制...');