<template>
    <div>
        <el-container>
            <el-aside style="width: 30%;">
                <setting @start-train="startTrain" @pause-train="pauseTrain"></setting>
                <log ref="refLog"></log>                
            </el-aside>
            <el-main>
                <process ref="refProcess"></process>  
            </el-main>
        </el-container>
    </div>
</template>

<script setup lang="ts">
import {ref, onMounted} from 'vue'
import Setting from './setting.vue'
import Log from './log.vue'
import Process from './process.vue'

const ws = new WebSocket('ws://10.10.65.82:8080/echo')
const refProcess = ref<any>(null)
const refLog = ref<any>(null)

onMounted(() => {
    ws.addEventListener('open', handleWsOpen.bind(this), false)
    ws.addEventListener('close', handleWsClose.bind(this), false)
    ws.addEventListener('error', handleWsError.bind(this), false)
    ws.addEventListener('message', handleWsMessage.bind(this), false)
})

const startTrain = (settingData: any) => {
    ws.send(JSON.stringify({type: 'trainProcess', setting: settingData}))
}

const pauseTrain = () => {
    ws.send(JSON.stringify({type: 'trainPause'}))
}

const handleWsOpen = () => {
    console.log('WebSocket2已经打开 ')
}
const handleWsClose = (e: any) => {
    console.log('WebSocket2关闭')
    console.log(e)
}
const handleWsError = (e:any) => {
    console.log('WebSocket2发生错误')
    console.log(e)
}
const handleWsMessage = (e: any) => {
    const result = JSON.parse(e.data)
    console.log('res:', result)
    if (result.log) refLog.value?.updateLog(result)
    if (result.status) return
    refProcess.value?.updateProcess(result)
}

</script>

<style scoped>

</style>
<style lang="scss">
.el-slider__bar {
    background-color: $study-theme;
}
.el-slider__button {
    width: 10px;
    height: 10px;
    border-color: $study-theme;
    background-color: $study-theme;
}
</style>