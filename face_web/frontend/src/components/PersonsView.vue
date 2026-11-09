<script setup>
import { ref, onMounted } from 'vue'

const API = 'http://127.0.0.1:8000'

const persons = ref([])
const keyword = ref('')
const selectedPerson = ref(null)
const photos = ref([])

// 新增表单
const showAddForm = ref(false)
const newPerson = ref({ student_no: '', name: '', dept: '' })

// 编辑表单
const showEditForm = ref(false)
const editPerson = ref({ name: '', dept: '' })

const message = ref('')

// 自定义确认弹窗（替代原生 confirm，避免在预览 iframe 中被禁用）
const confirmDialog = ref({ show: false, title: '', body: '', onOk: null })

async function loadPersons() {
  const url = keyword.value
    ? `${API}/api/persons?keyword=${encodeURIComponent(keyword.value)}`
    : `${API}/api/persons`
  const res = await fetch(url)
  const json = await res.json()
  persons.value = json.data
}

async function selectPerson(p) {
  selectedPerson.value = p
  await loadPhotos(p.id)
}

async function loadPhotos(personId) {
  const res = await fetch(`${API}/api/persons/${personId}/photos`)
  const json = await res.json()
  photos.value = json.data
}

async function addPerson() {
  const res = await fetch(`${API}/api/persons`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(newPerson.value)
  })
  if (res.ok) {
    message.value = `已新增 ${newPerson.value.name}`
    newPerson.value = { student_no: '', name: '', dept: '' }
    showAddForm.value = false
    loadPersons()
  } else {
    const err = await res.json()
    message.value = `新增失败：${err.detail || '未知错误'}`
  }
}

function openEdit(p) {
  // 记录编辑对象 id，避免依赖 selectedPerson 导致改错人
  editPerson.value = { id: p.id, name: p.name, dept: p.dept || '' }
  selectedPerson.value = p
  loadPhotos(p.id)
  showEditForm.value = true
}

async function saveEdit() {
  const res = await fetch(`${API}/api/persons/${editPerson.value.id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name: editPerson.value.name, dept: editPerson.value.dept })
  })
  if (res.ok) {
    message.value = '修改成功'
    showEditForm.value = false
    loadPersons()
    if (selectedPerson.value?.id === editPerson.value.id) {
      selectedPerson.value = { ...selectedPerson.value, name: editPerson.value.name, dept: editPerson.value.dept }
    }
  } else {
    const err = await res.json()
    message.value = `修改失败：${err.detail || '未知错误'}`
  }
}

async function deletePerson(p) {
  confirmDialog.value = {
    show: true,
    title: `删除人员「${p.name}」`,
    body: `将连带删除其 ${p.sample_count} 条特征、${p.photo_count} 张照片和全部考勤记录，不可恢复！`,
    onOk: async () => {
      const res = await fetch(`${API}/api/persons/${p.id}`, { method: 'DELETE' })
      if (res.ok) {
        message.value = `已删除 ${p.name}`
        if (selectedPerson.value?.id === p.id) {
          selectedPerson.value = null
          photos.value = []
        }
        loadPersons()
      } else {
        const err = await res.json()
        message.value = `删除失败：${err.detail || '未知错误'}`
      }
    }
  }
}

async function deletePhoto(ph) {
  confirmDialog.value = {
    show: true,
    title: `删除「${selectedPerson.value.name}」的一张样本`,
    body: '将同时删除该样本对应的特征，删除后请到 QT 客户端重新录入补充。',
    onOk: async () => {
      const res = await fetch(`${API}/api/persons/${selectedPerson.value.id}/photos/${ph.id}`, { method: 'DELETE' })
      if (res.ok) {
        message.value = '样本已删除'
        await loadPhotos(selectedPerson.value.id)
        loadPersons()
      } else {
        const err = await res.json()
        message.value = `删除失败：${err.detail || '未知错误'}`
      }
    }
  }
}

function closeConfirm() {
  confirmDialog.value = { show: false, title: '', body: '', onOk: null }
}

onMounted(loadPersons)
</script>

<template>
  <div>
    <!-- 工具栏 -->
    <div class="toolbar">
      <input
        v-model="keyword"
        type="text"
        placeholder="搜索学号或姓名"
        class="search-input"
        @input="loadPersons"
      />
      <button class="btn primary" @click="showAddForm = !showAddForm">
        {{ showAddForm ? '收起' : '+ 新增人员' }}
      </button>
    </div>

    <!-- 提示消息 -->
    <div v-if="message" class="message">{{ message }}</div>

    <!-- 新增表单 -->
    <div v-if="showAddForm" class="form-panel">
      <input v-model="newPerson.student_no" placeholder="学号（必填）" class="form-input" />
      <input v-model="newPerson.name" placeholder="姓名（必填）" class="form-input" />
      <input v-model="newPerson.dept" placeholder="院系（可选）" class="form-input" />
      <button class="btn primary" @click="addPerson">确认新增</button>
      <button class="btn" @click="showAddForm = false">取消</button>
      <span class="hint">新增后请到 QT 客户端录入人脸</span>
    </div>

    <!-- 编辑表单 -->
    <div v-if="showEditForm" class="form-panel">
      <div class="edit-row">
        <label class="edit-label">姓名</label>
        <input v-model="editPerson.name" placeholder="姓名" class="form-input" />
      </div>
      <div class="edit-row">
        <label class="edit-label">院系</label>
        <input v-model="editPerson.dept" placeholder="院系" class="form-input" />
      </div>
      <div class="edit-actions">
        <button class="btn primary" @click="saveEdit">保存修改</button>
        <button class="btn" @click="showEditForm = false">取消</button>
      </div>
      <div class="hint">人脸样本请在下方缩略图点 × 删除，删除后到 QT 客户端重新录入补充</div>
    </div>

    <!-- 人脸样本编辑区（编辑人员时显示） -->
    <div v-if="showEditForm" class="photo-edit-panel">
      <div class="photo-edit-title">人脸样本（点击 × 删除单张）</div>
      <div class="photo-grid editable">
        <div v-for="ph in photos" :key="ph.id" class="photo-item">
          <img
            v-if="ph.aligned_path"
            :src="`${API}/api/photos/${ph.aligned_path}`"
            :alt="`样本 ${ph.id}`"
          />
          <span v-else class="no-photo">无图</span>
          <button class="photo-del" title="删除该样本" @click="deletePhoto(ph)">×</button>
        </div>
      </div>
      <p v-if="!photos.length" class="empty">该人员暂无样本图片，请到 QT 客户端录入</p>
    </div>

    <div class="persons-layout">
      <!-- 人员列表 -->
      <div class="person-list">
        <div
          v-for="p in persons"
          :key="p.id"
          class="person-item"
          :class="{ active: selectedPerson?.id === p.id }"
          @click="selectPerson(p)"
        >
          <div class="row-head">
            <span class="name">{{ p.name }}</span>
            <span class="ops">
              <button class="mini-btn" @click.stop="openEdit(p)" title="修改">改</button>
              <button class="mini-btn danger" @click.stop="deletePerson(p)" title="删除">删</button>
            </span>
          </div>
          <div class="meta">{{ p.student_no }} · {{ p.dept || '未填院系' }}</div>
          <div class="badge">样本 {{ p.sample_count }} 张 · 照片 {{ p.photo_count }} 张</div>
        </div>
        <div v-if="!persons.length" class="empty">暂无人员</div>
      </div>

      <!-- 样本查看 -->
      <div class="photo-panel">
        <template v-if="selectedPerson">
          <h3>{{ selectedPerson.name }} 的人脸样本</h3>
          <div class="photo-grid">
            <div v-for="ph in photos" :key="ph.id" class="photo-item">
              <img
                v-if="ph.aligned_path"
                :src="`${API}/api/photos/${ph.aligned_path}`"
                :alt="`样本 ${ph.id}`"
              />
              <span v-else class="no-photo">无图</span>
            </div>
          </div>
          <p v-if="!photos.length" class="empty">该人员暂无样本图片</p>
        </template>
        <div v-else class="placeholder">← 点击左侧人员查看样本</div>
      </div>
    </div>

    <!-- 自定义确认弹窗 -->
    <div v-if="confirmDialog.show" class="modal-mask" @click.self="closeConfirm">
      <div class="modal">
        <div class="modal-title">{{ confirmDialog.title }}</div>
        <div class="modal-body">{{ confirmDialog.body }}</div>
        <div class="modal-actions">
          <button class="btn" @click="closeConfirm">取消</button>
          <button class="btn danger" @click="confirmDialog.onOk(); closeConfirm()">确认删除</button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.toolbar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; }
.search-input {
  flex: 1; max-width: 320px; padding: 8px 12px;
  border: 1px solid #ddd; border-radius: 6px; font-size: 14px;
}
.btn {
  padding: 8px 16px; border: 1px solid #ddd; background: #fff;
  border-radius: 6px; cursor: pointer; font-size: 14px; color: #333;
}
.btn.primary { background: #4a6cf7; color: #fff; border-color: #4a6cf7; }
.btn.primary:hover { background: #3a5ce0; }
.btn:hover { border-color: #4a6cf7; }

.message {
  padding: 10px 14px; background: #f0f6ff; color: #4a6cf7;
  border-radius: 6px; margin-bottom: 12px; font-size: 14px;
}

.form-panel {
  display: flex; gap: 8px; flex-wrap: wrap; align-items: center;
  padding: 14px; background: #fafafa; border: 1px solid #ececec;
  border-radius: 8px; margin-bottom: 16px;
}
.form-input {
  padding: 8px 12px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px;
}
.hint { color: #999; font-size: 12px; }

.persons-layout { display: flex; gap: 24px; }

.person-list { width: 320px; flex-shrink: 0; }
.person-item {
  padding: 14px; border: 1px solid #ececec; border-radius: 8px;
  margin-bottom: 10px; cursor: pointer; transition: all 0.15s;
}
.person-item:hover { border-color: #4a6cf7; }
.person-item.active { border-color: #4a6cf7; background: #f5f8ff; }
.row-head { display: flex; justify-content: space-between; align-items: center; }
.name { font-size: 15px; font-weight: 500; }
.ops { display: flex; gap: 6px; }
.mini-btn {
  padding: 2px 10px; font-size: 12px; border: 1px solid #ddd;
  border-radius: 4px; background: #fff; cursor: pointer; color: #333;
}
.mini-btn:hover { border-color: #4a6cf7; color: #4a6cf7; }
.mini-btn.danger:hover { border-color: #e74c3c; color: #e74c3c; }
.meta { font-size: 13px; color: #999; margin-top: 4px; }
.badge {
  display: inline-block; margin-top: 8px; font-size: 12px; color: #4a6cf7;
  background: #eef2ff; padding: 2px 8px; border-radius: 10px;
}

.photo-panel { flex: 1; }
.photo-panel h3 { font-size: 16px; margin-bottom: 16px; }
.photo-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(140px, 1fr)); gap: 12px; }
.photo-item {
  border: 1px solid #ececec; border-radius: 8px; overflow: hidden;
  aspect-ratio: 1; background: #fafafa;
}
.photo-item img { width: 100%; height: 100%; object-fit: cover; }
.no-photo { display: flex; height: 100%; align-items: center; justify-content: center; color: #ccc; }

.placeholder { color: #ccc; padding: 60px 0; text-align: center; }
.empty { color: #ccc; text-align: center; padding: 40px 0; }

.edit-row { display: flex; align-items: center; gap: 8px; }
.edit-label { width: 36px; font-size: 13px; color: #666; flex-shrink: 0; }
.edit-actions { display: flex; gap: 8px; }

.photo-edit-panel {
  padding: 14px; background: #fafafa; border: 1px solid #ececec;
  border-radius: 8px; margin-bottom: 16px;
}
.photo-edit-title { font-size: 14px; font-weight: 500; margin-bottom: 12px; color: #333; }
.photo-grid.editable .photo-item { position: relative; }
.photo-del {
  position: absolute; top: 4px; right: 4px;
  width: 24px; height: 24px; border-radius: 50%;
  border: none; background: rgba(231, 76, 60, 0.9); color: #fff;
  font-size: 16px; line-height: 1; cursor: pointer;
  display: flex; align-items: center; justify-content: center;
}
.photo-del:hover { background: #c0392b; }

.modal-mask {
  position: fixed; inset: 0; background: rgba(0, 0, 0, 0.4);
  display: flex; align-items: center; justify-content: center; z-index: 1000;
}
.modal {
  background: #fff; border-radius: 10px; padding: 24px;
  width: 360px; max-width: 90vw; box-shadow: 0 8px 30px rgba(0,0,0,0.2);
}
.modal-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #333; }
.modal-body { font-size: 14px; color: #555; margin-bottom: 20px; line-height: 1.6; }
.modal-actions { display: flex; justify-content: flex-end; gap: 10px; }
.btn.danger { background: #e74c3c; color: #fff; border-color: #e74c3c; }
.btn.danger:hover { background: #c0392b; }

</style>
