/**
* * Custom drag and drop commands for Cypress
* * Useful to handle edge cases where the `cypress-drag-drop` plugin could not work properly
* * Less feature rich than the plugin, but more reliable
*/

export const move = (
  draggableSubject: JQuery<HTMLElement>,
  dropSelector: string,
  dropX: number = 0,
  dropY: number = 0
) => {
  const draggedElement = draggableSubject.get()[0];

  return cy.window().then((window) => {
    cy.wrap(draggedElement).trigger('mousedown', { view: window, force: true });
    cy.get(dropSelector)
      .trigger('mousemove', dropX, dropY, { force: true, view: window })
      .trigger('mouseup', dropX, dropY, { force: true, view: window });
  });
};

export const drag = (
  draggableSubject: JQuery<HTMLElement>,
  dropSelector: string,
  dropX: number = 0,
  dropY: number = 0
) => {
  const dataTransfer = new DataTransfer();

  cy.wrap(draggableSubject)
    .trigger('dragstart', { dataTransfer })
    .trigger('dragleave', { dataTransfer });

  cy.get(dropSelector)
    .first()
    .trigger('dragenter', dropX, dropY, {
      dataTransfer,
      position: 'left',
      force: true,
    })
    .trigger('dragover', dropX, dropY, {
      dataTransfer,
      force: true,
      position: 'left',
    })
    .trigger('drop', dropX, dropY, {
      dataTransfer,
      position: 'left',
      force: true,
    })
    .trigger('dragend', {
      dataTransfer,
      force: true,
    });
};