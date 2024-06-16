const chartsRouter = {
    path: '/charts',
    component: () => import('@/layout/index.vue'),
    name: 'Charts',
    meta: {
        hidden: false,
        title: '图表',
        icon: 'Platform'
    },    
    children: [
        {
            path: '/charts/keyboard',
            component: () => import('@/views/profile/index.vue'),
            name: 'KeyboardChart',
            meta: {
                title: 'keyboard',
                hidden: false,
                icon: 'Platform'
            }
        },
        {
            path: '/charts/line',
            component: () => import('@/views/echarts/Line.vue'),
            name: 'LineChart',
            meta: {
                title: 'line',
                hidden: false,
                icon: 'Platform'
            }
        }
    ]    
}

export default chartsRouter