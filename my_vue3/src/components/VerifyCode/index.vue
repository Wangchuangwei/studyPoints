<template>
    <div class="canvas-box" :style="{height: contentHeight + 'px'}">
        <canvas
            id="id-canvas"
            class="id-canvas"
            :height="contentHeight"
            :width="contentWidth"
        ></canvas>
    </div>
</template>

<script setup lang="ts" name="Identify">
import {onMounted, watch} from 'vue'

interface IProps {
    identifyCode?: string
    fontSizeMin?: number
    fontSizeMax?: number
    backgroundColorMin?: number
    backgroundColorMax?: number
    dotColorMin?: number
    dotColorMax?: number
    contentWidth?: number
    contentHeight?: number
}

const props = withDefaults(defineProps<IProps>(), {
    identifyCode: '1234',
    fontSizeMin: 25,
    fontSizeMax: 35,
    backgroundColorMin: 200,
    backgroundColorMax: 220,
    dotColorMin: 60,
    dotColorMax: 120,
    contentWidth: 100,
    contentHeight: 40, 
})

const randomNum = (min: number, max: number) => {
    return Math.floor(Math.random() * (max - min) + min)
}

const randomColor = (min: number, max: number) => {
    let r = randomNum(min, max)
    let g = randomNum(min, max)
    let b = randomNum(min, max)
    return 'rgb(' + r + ',' + g + ',' + b + ')'
}

const drawPic = () => {
    let canvas = document.getElementById('id-canvas') as HTMLCanvasElement
    let ctx = canvas.getContext('2d') as CanvasRenderingContext2D
    ctx.textBaseline = 'bottom'
    ctx.fillStyle = '#e6ecfd'
    ctx.fillRect(0, 0, props.contentWidth, props.contentHeight)
    for (let i = 0; i < props.identifyCode.length; i++) {
        drawText(ctx, props.identifyCode[i], i)
    }
    drawLine(ctx)
    drawDot(ctx)
}

const drawText = (ctx: CanvasRenderingContext2D, txt: string, i: number) => {
    ctx.fillStyle = randomColor(50, 160)
    ctx.font = randomNum(props.fontSizeMin, props.fontSizeMax) + 'px SimHei'
    let x = (i + 1) * (props.contentWidth / (props.identifyCode.length + 1))
    let y = randomNum(props.fontSizeMax, props.contentHeight - 5)
    const deg = randomNum(-30, 30)
    ctx.translate(x, y)
    ctx.rotate((deg * Math.PI) / 180)
    ctx.fillText(txt, 0, 0)
    ctx.rotate((-deg * Math.PI) / 180)
    ctx.translate(-x, -y)
}

const drawLine = (ctx: CanvasRenderingContext2D) => {
    for (let i = 0; i < 4; i++) {
        ctx.strokeStyle = randomColor(100, 200)
        ctx.beginPath()
        ctx.moveTo(randomNum(0, props.contentWidth),randomNum(0, props.contentHeight))
        ctx.lineTo(randomNum(0, props.contentWidth),randomNum(0, props.contentHeight))
        ctx.stroke()
    }
}

const drawDot = (ctx: CanvasRenderingContext2D) => {
    for (let i = 0; i < 30; i++) {
        ctx.fillStyle = randomColor(0, 255)
        ctx.beginPath()
        ctx.arc(randomNum(0, props.contentWidth),randomNum(0, props.contentHeight),1,0,2 * Math.PI)
        ctx.fill()
    }
}

onMounted(() => {
    drawPic()
})

watch(
    () => props.identifyCode, 
    (value) => {
        console.log(value)
        drawPic()
    }
)

</script>

<style scoped lang="scss">
.canvas-box {
    cursor: pointer;
    .id-canvas {
        height: 100%;
    }
}
</style>