import {createRouter, createWebHashHistory} from 'vue-router'
import { constantRoute, studyRoute } from "./routes";

let router = createRouter({
    history: createWebHashHistory(),
    routes: [...constantRoute, ...studyRoute],

})

export default router