export interface DatasetMetadata {
  // TODO
}

export interface DatasetsListingFilters {
  page: number
  per_page: number
}

export interface Dataset {
  name: string
  description: string
  n_rows: number
  n_columns: number
  n_bytes: number
  split_target: string
  split_type: string
  split_actual: string
  metadata: DatasetMetadata
  created_at: Date
  updated_at: Date
}

export interface Paginated<T> {
  total: number,
  data: T[],
}

// A mock function to mimic makinggsan async request for data
export function fetchDatasets(_filters: DatasetsListingFilters): Promise<Paginated<Dataset>> {
  return new Promise((resolve) =>
    setTimeout(() => resolve({ total: 1, data: [{
      name: "asdasd",
      description: "loren loren",
      n_rows: 30,
      n_columns: 30,
      n_bytes: 256,
      metadata: {},
      created_at: new Date(),
      split_type: "scaffold",
      split_target: "60-20-20",
      split_actual: "58-22-20",
      updated_at: new Date(),
    }]}), 500)
  )
}
