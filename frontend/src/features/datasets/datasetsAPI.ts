interface User {
  fullName: string
  avatarUrl: string
  id: number
}
export interface DatasetMetadata {
  // TODO
}

export interface DatasetsListingFilters {
  page: number
  per_page: number
}

export interface Dataset {
  id: number;
  name: string
  description: string
  rows: number
  columns: number
  bytes: number
  splitType: string
  splitTarget:string
  splitActual:string
  metadata: DatasetMetadata,
  createdAt: Date,
  updatedAt: Date,
  createdBy: User
}

export interface Paginated<T> {
  total: number,
  data: T[],
}


export const mockDataset = () => 
  ({
    id: Math.ceil(Math.random() * (1 << 30)),
    name: 'NINK Dataset 2022',
    rows: 2000,
    columns: 3100,
    bytes: 3100,
    splitType: 'scaffold',
    description: 'asdasdasd',
    updatedAt: new Date(),
    metadata: {},
    splitActual: '59-21-20',
    splitTarget: '60-20-20',
    createdAt: new Date(),
    createdBy: {
      id: 1,
      fullName: 'Vitor',
      avatarUrl: 'https://chat.shawandpartners.com/avatar/jose.gilberto',
    }
  })
const layout = {
  align: 'center',
  headerAlign: 'center'
} as const

// A mock function to mimic makinggsan async request for data
export function fetchDatasets(_filters: DatasetsListingFilters): Promise<Paginated<Dataset>> {
  return new Promise((resolve) =>
    setTimeout(() => resolve({ total: 1, data: [mockDataset(), mockDataset()]}), 500)
  )
}
