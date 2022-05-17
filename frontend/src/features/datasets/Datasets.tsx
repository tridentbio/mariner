import { GridColDef, DataGrid } from '@mui/x-data-grid'
import { format } from 'date-fns'
import { Avatar, IconButton } from '@mui/material'
import { Dataset, mockDataset } from './datasetsAPI'
import ReadMoreIcon from '@mui/icons-material/ReadMore'
import DownloadIcon from '@mui/icons-material/Download'

/**
 * Sortable fields:
 * - Rows
 * - Columns
 * - Created at
 *
 * Filterable fields:
 * - Name
 */

const layout = { align: 'center', headerAlign: 'center' } as const
const mockedRows: Dataset[] = new Array(30).fill(0).map(mockDataset)

const Datasets = () => {
  const menu = (
    <>
      <IconButton>
        <ReadMoreIcon/>
      </IconButton>

      <IconButton>
        <DownloadIcon/>
      </IconButton>
    </>
  )

  const columns: GridColDef[] = [
    { field: 'name', headerName: 'NAME', width: 200 },
    { field: 'rows', headerName: 'ROWS' },
    { field: 'columns', headerName: 'COLUMNS' },
    { field: 'createdAt', headerName: 'CREATED AT', width: 120, valueFormatter: (params: {value: Date}) => format(params.value, 'dd-MM-yyyy') },
    { field: 'createdBy', headerName: 'CREATED BY', width: 120, renderCell: (params: { value?: Dataset['createdBy']}) => <Avatar src={params.value?.avatarUrl}/> },
    { field: 'actions', headerName: '', renderCell: () => menu }
  ].map(val => ({ ...layout, ...val }))
  return (
    <div style={{ height: 800 }}>
      <DataGrid rows={mockedRows} columns={columns}
        rowsPerPageOptions={[5, mockedRows.length, 100]}
        rowCount={mockedRows.length}
        pageSize={100}
        onSortModelChange={console.log}
        onFilterModelChange={console.log}
        paginationMode="server"
        onPageChange={console.log}
        onPageSizeChange={console.log}
      />
    </div>
  )
}

export default Datasets
