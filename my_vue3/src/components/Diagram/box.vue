<template>
    <div id="boxDiagram" style="height: 500px; width: 100%; margin-top: 20px"></div>
</template>

<script setup lang="ts">
import {onMounted, onBeforeUnmount} from 'vue';
import * as echarts from 'echarts';

type EChartsOption = echarts.EChartsOption;

let option: EChartsOption;
let myChart: any = null

const data = [
    [850, 740, 900, 1070, 930],
    [960, 940, 960, 940, 880],
    [880, 880, 880, 860, 720],
    [890, 810, 810, 820, 800],
    [890, 840, 780, 810, 760]
]

// const data1 = echarts.dataTool.prepareBoxplotData(data)

option = {
  title: [
    {
      text: '模型在指标上的盒图',
      left: 'center'
    },
  ],
  dataset: [
    {
      // prettier-ignore
      source: [
        [850, 740, 900, 1070, 930, 850, 950, 980, 980, 880, 1000, 980, 930, 650, 760, 810, 1000, 1000, 960, 960],
        [960, 940, 960, 940, 880, 800, 850, 880, 900, 840, 830, 790, 810, 880, 880, 830, 800, 790, 760, 800],
        [880, 880, 880, 860, 720, 720, 620, 860, 970, 950, 880, 910, 850, 870, 840, 840, 850, 840, 840, 840],
        [890, 810, 810, 820, 800, 770, 760, 740, 750, 760, 910, 920, 890, 860, 880, 720, 840, 850, 850, 780],
        [890, 840, 780, 810, 760, 810, 790, 810, 820, 850, 870, 870, 810, 740, 810, 940, 950, 800, 810, 870]
      ]
    },
    {
      transform: {
        type: 'boxplot',
        config: { itemNameFormatter: 'expr {value}' }
      }
    },
    {
      fromDatasetIndex: 1,
      fromTransformResult: 1
    }
  ],
  tooltip: {
    trigger: 'item',
    axisPointer: {
      type: 'shadow'
    }
  },
  grid: {
    left: '10%',
    right: '10%',
    bottom: '15%'
  },
  xAxis: {
    type: 'category',
    // data: ['1', '2', '3', '4', '5'],
    boundaryGap: true,
    nameGap: 30,
    splitArea: {
      show: false
    },
    splitLine: {
      show: false
    }
  },
  yAxis: {
    type: 'value',
    splitArea: {
      show: true
    }
  },
  series: [
    {
      name: 'boxplot',
      type: 'boxplot',
    //   data: data
      datasetIndex: 1
    }
  ]
};

const initChart = () => {
    const chartDom = document.getElementById('boxDiagram')!;
    myChart = echarts.init(chartDom);
    option && myChart.setOption(option);
}

onMounted(() => {
    initChart()
})

onBeforeUnmount(() => {
    if (!myChart) return
    myChart.dispose()
    myChart = null
})

</script>

<style scoped>

</style>