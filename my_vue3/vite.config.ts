import { ConfigEnv, UserConfigExport, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import path from 'path'
import {createSvgIconsPlugin} from 'vite-plugin-svg-icons'
import {viteMockServe} from 'vite-plugin-mock'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
// import monacoEditorPlugin from 'vite-plugin-monaco-editor'

export default (({command, mode}: ConfigEnv): UserConfigExport => {
  const env = loadEnv(mode, process.cwd())
  return {
    plugins: [
      vue(), 
      createSvgIconsPlugin({
        iconDirs: [path.resolve(process.cwd(), 'src/assets/icons')],
        symbolId: 'icon-[dir]-[name]',
      }),
      viteMockServe({
        localEnabled: command === 'serve',
      }),
      AutoImport({
        resolvers: [ElementPlusResolver()]
      }),
      Components({
        resolvers: [ElementPlusResolver()],
      }),
      // monacoEditorPlugin({
      //   languageWorkers: ['editorWorkerService', 'typescript', 'json', 'html']
      // })
    ],
    resolve:{
      alias:{
        "@":path.resolve(__dirname, 'src')
      }
    },
    css: {
      preprocessorOptions: {
        scss: {
          javascriptEnabled: true,
          additionalData: '@import "./src/styles/variable.scss";',
        }
      }
    },
    server: {
      proxy: {
        '/api': {   
          target: env.VITE_SERVE,
          changeOrigin: true,
        },
        '/remote': {
          target: 'http://10.10.65.82:3000',
          changeOrigin: true
        },
        '/expe': {
          target: 'http://10.10.65.82:8086',
          changeOrigin: true,
          rewrite: path => path.replace(RegExp(`^expe`), 'api')
        }
      }
    },
    // 强制预构建插件包
    // optimizeDeps: {
    //   include: [
    //     `monaco-editor/esm/vs/language/json/json.worker`,
    //     `monaco-editor/esm/vs/language/typescript/ts.worker`,
    //     `monaco-editor/esm/vs/editor/editor.worker`
    //   ], 
    // },
    // transpileDependencies: true,
    // configureWebpack: {
    //   plugins: [],
    // }
  }
})
