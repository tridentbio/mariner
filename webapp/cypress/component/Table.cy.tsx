import { store } from "@app/store";
import { Experiment } from "@app/types/domain/experiments";
import Table from "@components/templates/Table";
import { Column, State } from "@components/templates/Table/types";
import { ThemeProvider } from "@mui/material";
import { Provider } from "react-redux";
import { theme } from "theme";
import { columns, rows } from '../../tests/fixtures/table/experimentsDataMock';
import { NonUndefined } from "@utils";

describe('DataPreprocessingInput.cy.tsx', () => {
  let state: State

  const MountedComponent = () => {
    return (
      <Provider store={store}>
        <ThemeProvider theme={theme}>
          <Table
            columns={columns}
            rows={rows as Experiment[]}
            rowKey={row => row.id}
            onStateChange={newState => state = newState}
          />
        </ThemeProvider>
      </Provider>
    );
  }

  const closePopover = () => cy.get('.MuiBackdrop-root').click()

  const toggleDisplayedColumns = (columns: Column<any, any>[], {check}: {check: boolean}) => {
    cy.get('[data-testid="column-picker-btn"]').click()

    columns.forEach(column => {
      cy.get('li[role="treeitem"]').contains(column.name).parent().find('input[type="checkbox"]')[check ? 'check' : 'uncheck']()
    })

    closePopover()
  }

  it('should reorder the columns', () => {
    cy.mount(<MountedComponent />)

    columns.forEach((column, index) => {
      const lastCol = index + 1 >= columns.length

      if(lastCol) return

      cy.get(`[data-testid="draggable-cell-${index}"]`).drag(`[data-testid="draggable-cell-${index + 1}"]`, {waitForAnimations: false})
    })

    //? Assert that all non fixed columns have a different position than before
    cy.get('[data-testcellname]').then((colElements) => {
      const newColNameOrder = Array.from(colElements).map((element) => element.getAttribute('data-testcellname'))

      columns.forEach((column, oldColumnIndex) => {
        if(!column.fixed) {
          expect(column.name).to.not.be.equal(newColNameOrder[oldColumnIndex])
        } else {
          expect(column.name).to.be.equal(newColNameOrder[oldColumnIndex])
        }
      })
    })
  })

  it('should sort the columns by clicking on the sorting button', () => {
    cy.mount(<MountedComponent />)

    const sortableColumns = columns.filter(column => column.sortable)

    sortableColumns.forEach((column, index) => {
      cy.get(`[data-testid="sorting-button-${column.name}"]`).click()

      cy.wait(200)

      cy.get(`[data-testid="sort-desc-${column.name}"]`).click()

      cy.get(`[data-testid="chip-sort-${column.name}"]`).should('exist')
    })

    cy.wait(50).then(() => {
      sortableColumns.forEach((column) => {
        const columnState = state?.sortModel.find(sortModel => sortModel.field === column.field)

        expect(columnState).to.not.be.undefined
        expect(columnState?.sort).to.be.equal('desc')
      })
  
      cy.get('[data-test-row-id]').then((rowElements) => {
        const newRowsOrder = Array.from(rowElements).map((element) => element.getAttribute('data-test-row-id'))
  
        //? Assert that the rows are being listed in decreasing order
        newRowsOrder.forEach((rowTestId, index) => {
          if(index + 1 >= newRowsOrder.length) return
  
          const nextRowId = newRowsOrder[index + 1]
  
          if(!rowTestId || !nextRowId) throw new Error('Row id not found')
  
          expect(parseInt(rowTestId)).to.be.greaterThan(parseInt(nextRowId))
        })
      })
    })
  })

  it('should hide and show columns by the column picker', () => {
    cy.mount(<MountedComponent />)

    const someNonFixedColumns = columns.filter(column => !column.fixed).slice(0, 3)

    toggleDisplayedColumns(someNonFixedColumns, {check: false})

    someNonFixedColumns.forEach(column => {
      cy.get(`[data-testcellname="${column.name}"]`).should('not.exist')
    })

    toggleDisplayedColumns(someNonFixedColumns, {check: true})

    someNonFixedColumns.forEach(column => {
      cy.get(`[data-testcellname="${column.name}"]`).should('exist')
    })
  })

  it('should filter the rows by the filter input', () => {
    cy.mount(<MountedComponent />)

    cy.get('[data-testid="add-filter-btn"]').click()

    const filterableColumns = columns.filter(column => !!column.filterSchema)

    //? Apply a filter to the filterable columns
    filterableColumns.forEach(column => {
      const filterTypes = column.filterSchema
      ? (Object.keys(column.filterSchema) as
          | (keyof NonUndefined<(typeof column)['filterSchema']>)[]
          | null)
      : null;

      cy.get(`[data-testid="add-filter-${column.name}-option"]`).click()

      filterTypes?.forEach(filterType => {
        const filter = cy.get(`[data-testid="filter-${column.name}"]`)

        switch(filterType) {
          case 'byContains':
            filter.click().get('li[role="option"]').first().click();
            break;
          default:
            filter.get('input[type="text"]').type('1');
            break
        }
      })
      
      cy.get(`[data-testid="add-filter-${column.name}-btn"]`).click()
      cy.get(`[data-testid="chip-filter-${column.name}"]`).should('exist')
    })

    closePopover()

    const clearAllFiltersBtn = cy.get('[data-testid="clear-all-filters-btn"]')

    //? Validate filters
    cy.get('[data-test-row-id]').then((rowElements) => {
      expect(rowElements.length).to.not.be.equal(rows.length)

      clearAllFiltersBtn.should('exist')

      state.filterModel?.items?.forEach?.((filterItem, index) => {
        const foundColumn = columns.find(column => column.field === filterItem.columnName)
        
        expect(foundColumn).not.to.be.undefined
      })
    })

    //? Test filters clear
    clearAllFiltersBtn.click()

    cy.get('[data-test-row-id]').then((rowElements) => {
      expect(rowElements.length).to.be.equal(rows.length)
    })
  })
})