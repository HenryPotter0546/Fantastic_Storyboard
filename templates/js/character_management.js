// 主角形象管理模块
let selectedCharacter = null;
let availableCharacters = [];
let currentCategory = '玄幻'; // 默认显示玄幻分类

// 显示主角形象选择区域
function showCharacterSelection() {
    document.getElementById('characterSelectionArea').style.display = 'block';
    document.getElementById('confirmCharacterBtn').disabled = true;
    selectedCharacter = null;
    
    // 重新绑定标签栏事件
    setTimeout(() => {
        document.querySelectorAll('.tab-item').forEach(tab => {
            // 移除旧的事件监听器
            tab.removeEventListener('click', tabClickHandler);
            // 添加新的事件监听器
            tab.addEventListener('click', tabClickHandler);
        });
    }, 50);
    
    // 默认加载玄幻分类的形象
    loadCharactersByCategory('玄幻');
    updateTabActive('玄幻');
}

// 标签点击处理函数
function tabClickHandler() {
    const category = this.getAttribute('data-category');
    if (category) {
        console.log('点击分类:', category); // 调试日志
        loadCharactersByCategory(category);
        updateTabActive(category);
    }
}

// 切换分类函数（用于内联事件）
function switchCategory(category) {
    console.log('切换分类:', category); // 调试日志
    loadCharactersByCategory(category);
    updateTabActive(category);
}

// 隐藏主角形象选择区域
function hideCharacterSelection() {
    document.getElementById('characterSelectionArea').style.display = 'none';
}

// 更新标签栏的激活状态
function updateTabActive(category) {
    // 移除所有标签的active类
    document.querySelectorAll('.tab-item').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // 为当前分类的标签添加active类
    const activeTab = document.querySelector(`[data-category="${category}"]`);
    if (activeTab) {
        activeTab.classList.add('active');
    }
}

// 加载指定分类的主角形象
async function loadCharactersByCategory(category) {
    try {
        console.log('正在加载分类:', category); // 调试日志
        const response = await fetch(`/available-characters?category=${encodeURIComponent(category)}`);
        console.log('API响应状态:', response.status); // 调试日志
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log('API返回数据:', data); // 调试日志
        availableCharacters = data.characters || [];
        
        // 更新当前分类
        currentCategory = category;
        
        // 渲染形象网格
        renderCharacterGrid();
        
    } catch (error) {
        console.error('获取主角形象列表失败:', error);
        availableCharacters = [];
        renderCharacterGrid();
    }
}

// 渲染形象网格
function renderCharacterGrid() {
    const gridContainer = document.getElementById('characterGrid');
    gridContainer.innerHTML = '';
    
    console.log('渲染形象网格，可用形象数量:', availableCharacters.length); // 调试日志
    
    if (availableCharacters.length === 0) {
        gridContainer.innerHTML = '<div style="text-align: center; color: #6c757d; padding: 20px;">该分类下暂无形象</div>';
        return;
    }
    
    availableCharacters.forEach(character => {
        console.log('渲染形象:', character); // 调试日志
        const characterItem = document.createElement('div');
        characterItem.className = 'character-item';
        characterItem.setAttribute('data-filename', character.filename);
        
        const imagePath = character.path || `${character.category}/${character.filename}`;
        console.log('图片路径:', `/static/character_images/${imagePath}`); // 调试日志
        
        characterItem.innerHTML = `
            <img src="/static/character_images/${imagePath}" alt="主角形象" 
                 onerror="this.src='/static/placeholder.png'; console.log('图片加载失败:', this.src);">
        `;
        
        // 添加点击事件
        characterItem.addEventListener('click', () => {
            selectCharacter(character, characterItem);
        });
        
        gridContainer.appendChild(characterItem);
    });
}

// 选择主角形象
function selectCharacter(character, element) {
    // 移除所有选中状态
    document.querySelectorAll('.character-item').forEach(item => {
        item.classList.remove('selected');
    });
    
    // 添加选中状态
    element.classList.add('selected');
    selectedCharacter = character;
    
    // 启用确认按钮
    document.getElementById('confirmCharacterBtn').disabled = false;
}

// 确认选择主角形象
async function confirmCharacter() {
    if (!selectedCharacter) {
        showCharacterStatus('请先选择一个主角形象', 'error');
        return;
    }
    
    try {
        const response = await fetch('/set-character', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({character_name: selectedCharacter.filename})
        });
        
        const result = await response.json();
        if (result.success) {
            showCharacterStatus('主角形象设置成功！', 'success');
            showCurrentCharacterInfo(selectedCharacter);
            setTimeout(hideCharacterStatus, 3000);
        } else {
            showCharacterStatus(`主角形象设置失败: ${result.message}`, 'error');
        }
    } catch (error) {
        showCharacterStatus(`设置请求失败: ${error.message}`, 'error');
    }
}

// 检查主角形象状态
async function checkCharacterStatus() {
    try {
        const response = await fetch('/character-status');
        const data = await response.json();
        
        if (data.selected_character) {
            selectedCharacter = data.selected_character;
            showCurrentCharacterInfo(data.selected_character);
        } else {
            showNoCharacterSelected();
        }
    } catch (error) {
        console.error('获取主角形象状态失败:', error);
        showNoCharacterSelected();
    }
}

// 显示当前主角形象信息
function showCurrentCharacterInfo(character) {
    const imageElement = document.getElementById('currentCharacterImage');
    const textElement = document.getElementById('currentCharacterText');
    
    // 使用完整路径显示图片
    const imagePath = character.path || `${character.category}/${character.filename}`;
    imageElement.src = `/static/character_images/${imagePath}`;
    imageElement.style.display = 'block';
    textElement.textContent = ''; // 不显示任何文字
    
    document.getElementById('currentCharacterInfo').style.display = 'block';
    document.getElementById('characterSelectionArea').style.display = 'none';
}

// 显示未选择主角形象状态
function showNoCharacterSelected() {
    const imageElement = document.getElementById('currentCharacterImage');
    const textElement = document.getElementById('currentCharacterText');
    
    imageElement.src = '';
    imageElement.style.display = 'none';
    textElement.textContent = ''; // 不显示任何文字
    
    document.getElementById('currentCharacterInfo').style.display = 'block';
    document.getElementById('characterSelectionArea').style.display = 'none';
}

// 显示主角形象状态消息
function showCharacterStatus(message, type) {
    const statusDiv = document.getElementById('characterStatus');
    statusDiv.style.display = 'block';
    statusDiv.textContent = message;
    statusDiv.className = `message-box ${type}`;
}

// 隐藏主角形象状态消息
function hideCharacterStatus() {
    document.getElementById('characterStatus').style.display = 'none';
}

// 初始化主角形象管理模块
function initCharacterManagement() {
    // 延迟绑定事件，确保DOM完全加载
    setTimeout(() => {
        // 为标签栏添加点击事件
        document.querySelectorAll('.tab-item').forEach(tab => {
            tab.addEventListener('click', tabClickHandler);
        });
    }, 100);
}

// 导出函数供外部使用
window.characterManagement = {
    loadCharactersByCategory,
    checkCharacterStatus,
    showCharacterSelection,
    hideCharacterSelection,
    confirmCharacter,
    initCharacterManagement
};

// 将切换分类函数暴露到全局作用域
window.switchCategory = switchCategory; 