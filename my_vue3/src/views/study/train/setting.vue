<template>
    <div class="title">参数选择</div>
    <div class="block-flex" style="margin-bottom: 15px">
        <div style="display: inline-block">
            <div class="setting-title">模型类型</div>
            <el-radio-group v-model="radio">
                <el-radio label="generation">提交生成</el-radio>
                <el-radio label="prediction">评审预测</el-radio>
            </el-radio-group>
        </div>
        <div style="display: inline-block">
            <div class="setting-title">选择模型</div>
            <el-select v-model="model" placeholder="Select">
                <el-option
                    v-for="item in options"
                    :key="item"
                    :label="item"
                    :value="item"
                />
            </el-select>
        </div> 
    </div>

    <el-card>
        <div class="setting-content">
            <div class="setting-title">Batch Size: {{batchSize}}</div>
            <el-slider v-model="batchSize" :min="1" :max="32" />           
        </div>
        <div class="setting-content">
            <div class="setting-title">输入长度:{{attention}}</div>
            <el-slider v-model="attention" :min="1" :max="512" />           
        </div>
        <div class="setting-content">
            <div class="setting-title">随机种子:{{demension}}</div>
            <el-slider v-model="demension" :min="1" :max="128"/>
        </div>
        <div class="setting-content">
            <div class="setting-title">学习率</div>
            <el-input v-model="learningRate" type="number" step="0.00001" />
        </div>
        <div class="setting-content">
            <div class="setting-title">训练步数</div>
            <el-input v-model="epoch" type="number" />
        </div>
        <el-button @click="startTrain">开始训练</el-button>
        <el-button @click="pauseTrain">停止训练</el-button>
    </el-card>
</template>

<script setup lang="ts">
import { ref, reactive, watch } from 'vue'

const radio = ref('generation')
const model = ref('Mucha')
let options = ref<Array<string>>(['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha'])

const batchSize = ref(8)
const attention = ref(256)
const demension = ref(42)
const learningRate = ref(0.00002)
const epoch = ref(12)

watch(
    () => radio.value, 
    (newValue) => {
        if (newValue == 'generation') options.value = ['CODISUM', 'CoreGen', 'CCRep', 'CodeBert', 'UniXcoder', 'Mucha']
        else options.value = ['CodeBert', 'UniXcoder', 'Mucha']
    }
)

const emits = defineEmits(['start-train', 'pause-train'])

const startTrain = () => {
    const settingData = {
        task: radio.value,
        model: model.value,
        batchSize: batchSize.value,
        attention: attention.value,
        demension: demension.value,
        lr: learningRate.value,
        epoch: epoch.value
    }
    emits('start-train', settingData)
}

const pauseTrain = () => {
    emits('pause-train')
}
</script>

<style scoped>
.setting-content {
    width: 250px;
    margin-bottom: 15px;
}
.block-flex {
    display: flex;
    align-items: center;
    justify-content: space-between;
}
</style>