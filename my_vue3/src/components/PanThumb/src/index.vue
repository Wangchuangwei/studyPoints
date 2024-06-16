<template>
    <div :style="{zIndex:zIndex,height:height,width:width}" class="pan-item">
        <div class="pan-info absolute_center">
            <svg viewBox="0 0 196 196">
                <circle class="circulo" r="96" cx="98" cy="98"></circle>
            </svg>
            <div :style="{height:textHeight,width:textWidth}" class="pan-info-roles-container absolute_center">
                <slot name="content"/>
            </div>
        </div>
        <div :style="{backgroundImage: `url(${image})`}" class="pan-thumb absolute_center"></div>
    </div>
</template>

<script setup lang="ts" name="PanThumb">
import {ref, computed} from 'vue'
const props = defineProps({
    image: {
        type: String,
        default: '/logo.gif',
        require: true
    },
    zIndex: {
        type: Number,
        default: 1
    },
    width: {
        type: String,
        default: '120px',
    },       
    height: {
        type: String,
        default: '120px',
    }, 
})
const textWidth: ref<string>= computed(() => {
    return Math.ceil(Number(props.width.substring(0, props.width.length - 2)) * 3 / 4) + 'px'
})
const textHeight: ref<string>= computed(() => {
    return Math.ceil(Number(props.height.substring(0, props.height.length - 2)) * 3 / 4) + 'px'
})
</script>

<style scoped lang="scss">
.pan-item {
    position: relative;
    border: 3px solid #dfe6ec;
    border-radius: 50%;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
    .pan-info {
        width: inherit;
        height: inherit;
        padding: 6px;
        text-align: center;
        overflow: hidden;
        border-radius: 50%;
        box-shadow: inset 0 0 0 5px rgba(0, 0, 0, 0.05);
        .circulo {
            stroke: #F85B5B;
            stroke-width: 5;
            stroke-dasharray: 625;
            stroke-linecap: round;
            fill: transparent;
            transform-origin: center center;
            transform: rotate(-90deg);
        }
        .pan-info-roles-container {
            border-radius: 50%;
            border: 2px dashed #b2daff;
        }
    }
    .pan-thumb {
        width: inherit;
        height: inherit;
        border-radius: 50%;
        background-position: center center;
        background-size: cover;
        overflow: hidden;
        transition: all 0.3s ease-in-out;
    }
    &:hover{
        .pan-thumb {
            transform: rotate(-110deg);
        }
        .pan-info .circulo {
            animation: roda 6s linear infinite;
        }
    } 
}
</style>