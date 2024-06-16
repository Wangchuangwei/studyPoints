import request from '@/utils/request'
import type {ExportTableResponseData} from '../type/index'

enum API {
    ALLROLE_URL = '/api/download/studyExport/',
    ALLDATASET_URL = '/api/download/dataset/',
    ADDROLE_URL = '/admin/acl/role/save',
    UPDATEROLE_URL = '/admin/acl/role/update',
    ALLPERMISSION_URL = '/admin/acl/permission/toAssign/',
    SETPERMISSION_URL = '/admin/acl/permission/doAssign/?',
    REMOVEROLE_URL = '/admin/acl/role/remove/',
  }

export const reqAllFileList = (page: number, limit: number, fileName: string) =>
    request.get<any, ExportTableResponseData>(
        API.ALLROLE_URL + `${page}/${limit}/?fileName=${fileName}`,
    )

export const reqAllDatasetList = () => request.get<any, ExportTableResponseData>(API.ALLDATASET_URL)