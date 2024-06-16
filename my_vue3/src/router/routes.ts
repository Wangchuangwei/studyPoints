/**
 * hidden: 是否为menu
 * icon: 图标
 */

import chartsRouter from './modules/charts'
import diffsRouter from './modules/diffs'
import downloadsRoutes from './modules/downloads'

export const constantRoute = [
    {
        path: '/login',
        component: () => import('@/views/login/index.vue'),
        name: 'login',
        meta: {
            title: 'login',
            hidden: true
        }
    },
    {
        path: '/',
        component: () => import('@/layout/index.vue'),
        name: 'layout',
        meta: {
            title: '',
            hidden: false,
        },
        redirect: '/home',
        children: [
            {
                path: '/home',
                component: () => import('@/views/home/index.vue'),
                meta: {
                    title: '首页',
                    hidden: false,
                    icon: 'HomeFilled'
                }
            }
        ]
    },
    {
        path: '/screen',
        component: () => import('@/views/screen/index.vue'),
        meta: {
            title: 'Screen',
            hidden: false,
            icon: 'Platform'
        },
        redirect: '/studyMain',
    },
    {
        path: '/404',
        component: () => import('@/views/404/index.vue'),
        name: '404',
        meta: {
            title: '404',
            hidden: true,
        }
    },
    {
        path: '/profile',
        component: () => import('@/layout/index.vue'),
        redirect: '/profile/index',
        meta: {
            title: '',
            hidden: true,
        },
        children: [
            {
                path: '/profile/index',
                component: () => import('@/views/profile/index.vue'),
                meta: {
                    title: 'Profile',
                    hidden: false,
                    icon: 'Platform'
                }
            }
        ]
    }
]

export const asyncRoute = [
    chartsRouter,
    diffsRouter,
    downloadsRoutes,
    {
        path: '/file',
        component: () => import('@/layout/index.vue'),
        name: 'File',
        redirect: '/file/index',
        meta: {
            title: '文件管理',
            icon: 'Platform',
            hidden: false,
        },
        children: [
            {
                path: '/file/index',
                component: () => import('@/views/project/index.vue'),
                name: 'FileList',
                meta: {
                    title: '文件列表',
                    hidden: false,
                    icon: 'Platform'
                }
            },
            {
                path: '/file/upload',
                component: () => import('@/views/project/uploadFile.vue'),
                name: 'UploadFile',
                meta: {
                    title: '上传文件',
                    hidden: false,
                    icon: 'Platform'
                }
            }
        ]
    },
    {
        path: '/acl',
        component: () => import('@/layout/index.vue'),
        name: 'Acl',
        meta: {
            title: '权限管理',
            hidden: false,
            icon: 'Lock'
        }, 
        redirect: '/acl/user',
        children: [
            {
                path: '/acl/user',
                component: () => import('@/views/acl/user/index.vue'),
                name: 'User',
                meta: {
                    title: '用户管理',
                    hidden: false,
                    icon: 'User'
                }
            }
        ]
    },
    {
        path: '/product',
        component: () => import('@/layout/index.vue'),
        name: 'Product',
        meta: {
            title: '商品管理',
            hidden: false,
            icon: 'Goods'
        },
        redirect: '/acl/user',
        children: [
            {
                path: '/acl/user',
                component: () => import('@/views/acl/user/index.vue'),
                name: 'User',
                meta: {
                    title: '权限管理',
                    hidden: false,
                    icon: 'User'
                }
            }
        ]
    }
]

export const studyRoute = [
    {
        path: '/studyHome',
        component: () => import('@/views/screen/index.vue'),
        meta: {
            title: '',
            hidden: true,
        },
        children: [
            {
                path: '/studyMain',
                // component: () => import('@/views/export/StudyFile.vue'),
                component: () => import('@/views/profile/index.vue'),
                meta: {
                    title: '主页',
                    hidden: false,
                    icon: 'home'
                }
            },
            {
                path: '/datasetAnalysis',
                component: () => import('@/views/study/dataset/index.vue'),
                meta: {
                    title: '数据集分析',
                    hidden: false,
                    icon: 'dataset'
                }
            },
            {
                path: '/Diff',
                component: () => import('@/views/study/diff/index.vue'),
                meta: {
                    title: '变更差异分析',
                    hidden: false,
                    icon: 'ast'
                }
            },
            {
                path: '/modelTrain',
                component: () => import('@/views/study/train/index.vue'),
                meta: {
                    title: '模型训练',
                    hidden: false,
                    icon: 'train'
                }
            },
            {
                path: '/resultAnalysis',
                component: () => import('@/views/study/result/index.vue'),
                meta: {
                    title: '结果分析',
                    hidden: false,
                    icon: 'result'
                },
                // redirect: '/resultAnalysis/generation',
                // children: [
                //     {
                //         path: '/resultAnalysis/generation',
                //         component: () => import('@/views/study/result/index.vue'),
                //         meta: {
                //             title: '提交生成',
                //             hidden: false,
                //             icon: 'result'
                //         }
                //     },
                //     {
                //         path: '/resultAnalysis/prediction',
                //         component: () => import('@/views/study/result/index.vue'),
                //         meta: {
                //             title: '代码评审',
                //             hidden: false,
                //             icon: 'result'
                //         }
                //     }
                // ]
            },            
        ]
    }

]

export const anyRoute = {
    path: '/:pathMatch(.*)*',
    redirect: '/404',
    name: 'Any',
    meta: {
        title: '任意路由',
        hidden: true
    }
}