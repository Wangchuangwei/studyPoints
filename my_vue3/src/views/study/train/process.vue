<template>
    <div class="title">训练进度</div>
    <div class="train_process">
        <div>Epoch{{currentEpoch}}:</div>
        <div>{{currentNum}}/{{total}}</div>
        <div>当前批次样本损失为：{{currentLoss}}</div>
    </div>
    <div class="train_speed">
        <div>{{currentSpeed}}</div>
        <el-progress :percentage="percentage" :color="customColor" :show-text="false"></el-progress>
    </div>
    <el-divider/>
    <div class="title">训练结果</div>
    <el-table :data="trainData" border style="width:100%" max-height="500px">
        <el-table-column prop="id"/>
        <el-table-column prop="epoch" label="epoch"/>
        <el-table-column prop="train_loss" label="train_loss"/>
        <el-table-column prop="val_loss" label="val_loss"/>
    </el-table>
    <el-divider/>
    <div class="download_model">
        <div>当前最优模型为<span>Epoch {{checkpoint}}</span></div>
        <el-button @click="downloadModel">下载模型</el-button>
    </div>
    <div class="show_result" style="margin-bottom: 0" v-if="download">Epoch{{checkpoint}}模型下载完成！</div>
</template>

<script setup lang="ts">
import {ref, onMounted} from 'vue'

const customColor = ref('#ff4b4b')
const download = ref(false)

const currentEpoch = ref(5)
let currentSpeed = ref('35% 4631/13232 [1:29:07<1:47:22, 1.12it/s]')
const currentNum = ref(4631)
const currentLoss = ref(3.3621)
const total = ref(13232)
const percentage = ref(35)
let trainData = ref<any>([
    {id: 0, epoch: 1, train_loss: 4.5157, val_loss: 3.5324},
    {id: 1, epoch: 2, train_loss: 3.7120, val_loss: 3.2451},
    {id: 2, epoch: 3, train_loss: 3.4255, val_loss: 3.1038},
    {id: 3, epoch: 4, train_loss: 3.2564, val_loss: 3.0153},

])
const checkpoint = ref(4)

const initProcess = () => {
    currentEpoch.value = 0
    currentSpeed.value = ''
    currentNum.value = 0
    currentLoss.value = 0
    total.value = 0
    percentage.value = 0
}

const getCheckPoint = () => {
    const length = trainData.value.length
    checkpoint.value = trainData.value[length - 1].epoch
}

const downloadModel = () => {
    if (trainData.value.length === 0) return
    download.value = true
}

onMounted(() => {
    // initProcess()
})

const updateProcess = (result: any) => {
    currentEpoch.value = result.epoch
    currentNum.value = result.sample
    total.value = result.total
    currentLoss.value = result.loss
    currentSpeed.value = result.speed
    percentage.value = result.percentage
    if (result.percentage == 100) {
        const length = trainData.value.length
        if (result.type == 'train') {
            trainData.value.push({
                id: length,
                epoch: result.epoch,
                train_loss: result.loss,
                val_loss: ''
            })
        } else {
            trainData.value[length - 1].val_loss = result.loss
            getCheckPoint()
        }
    }
}

defineExpose({
    updateProcess
})
</script>

<style scoped lang="scss">
.train_process {
    display: flex;
    justify-content: space-between;
    font-weight: 600;
    padding-right: 40px;
    margin-bottom: 20px;
}
.train_speed {
    div {
        color: #918a87;
        margin-bottom: 10px;
    }
}
.download_model {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-right: 40px;
    margin-bottom: 20px;
    span {
        margin-left: 10px;
        font-weight: 600;
    }
}

</style>