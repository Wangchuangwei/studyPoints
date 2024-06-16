function createProjectList() {
    return [
        {
            userId: 1,
            token: 'Admin Token',
            project: [
                { id: '1', label: '节点1', children: 
                    [
                        { id: '1-1', label: '节点1-1', children: [], isLeaf: true },
                        { id: '1-2', label: '节点1-2', children: 
                            [
                                { id: '1-2-1', label: '节点1-2-1', children: 
                                    [
                                        { id: '1-2-1-1', label: '节点1-2-1-1', children: 
                                            [
                                                { id: '1-2-1-1-1', label: '节点1-2-1-1-1', children: [], isLeaf: true },
                                            ], 
                                            isLeaf: false 
                                        },
                                        { id: '1-2-1-2', label: '节点1-2-1-2', children: [], isLeaf: true },
                                    ], 
                                    isLeaf: false 
                                },
                            ], 
                            isLeaf: false 
                        },
                        { id: '1-3', label: '节点1-3', children: [], isLeaf: true },
                    ], 
                    isLeaf: false 
                },
                { id: '2', label: '节点2', children: [], isLeaf: true },
                { id: '3', label: '节点3', children: 
                    [
                        { id: '3-1', label: '节点3-1', children: [], isLeaf: true },
                        { id: '3-2', label: '节点3-2', children: [], isLeaf: true },
                    ], 
                    isLeaf: false 
                },
            ]
        }
    ]
}

function getChildData(project, pre_id, ids) {
    if (ids.length === 0) {
        return project
    }
    const id = ids.shift()
    const cur_id = pre_id == ''? id : pre_id + '-' + id
    for(let i = 0; i < project.length; i++) {
        const item = project[i]
        if (item.id === cur_id) {
            return getChildData(item.children, cur_id, ids)
        }
    }
    return null
}

export {createProjectList, getChildData}