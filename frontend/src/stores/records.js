import { ref } from 'vue'
import { defineStore } from 'pinia'

import {
  createRecord,
  deleteRecord,
  listRecords,
  updateRecord as updateRecordApi,
} from '../api/records'

export const STATUS_OPTIONS = ['待投', '已投', '笔试', '面试', '挂', 'offer']

export const useRecordsStore = defineStore('records', () => {
  const records = ref([])
  const status = ref('')
  const loading = ref(false)

  async function fetchRecords(nextStatus = status.value) {
    status.value = nextStatus
    loading.value = true
    try {
      records.value = await listRecords(nextStatus)
    } finally {
      loading.value = false
    }
  }

  async function addRecord(payload) {
    const created = await createRecord(payload)
    records.value.unshift({ ...payload, record_id: created.record_id })
    return created
  }

  async function removeRecord(recordId) {
    await deleteRecord(recordId)
    records.value = records.value.filter((item) => item.record_id !== recordId)
  }

  async function updateRecord(recordId, payload) {
    const updated = await updateRecordApi(recordId, payload)
    const index = records.value.findIndex((item) => item.record_id === recordId)
    if (index >= 0) {
      if (status.value && updated.status !== status.value) {
        records.value.splice(index, 1)
      } else {
        records.value[index] = updated
      }
    }
    return updated
  }

  return {
    records,
    status,
    loading,
    fetchRecords,
    addRecord,
    updateRecord,
    removeRecord,
  }
})
