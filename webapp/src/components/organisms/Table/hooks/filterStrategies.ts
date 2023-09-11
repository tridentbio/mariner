export interface FilterStrategy<T> {
  validate(value: T): boolean;
}

export class EqualsFilterStrategy implements FilterStrategy<string | number> {
  constructor(private targetName: string) {}

  validate(value: string | number): boolean {
    return value == this.targetName;
  }
}

export class ContainsFilterStrategy implements FilterStrategy<string | number> {
  constructor(private options: (string | number)[]) {}

  validate(value: string | number): boolean {
    return this.options.some((option) => option === value);
  }
}

export class IncludesFilterStrategy implements FilterStrategy<string> {
  constructor(private targetName: string) {}

  validate(value: string): boolean {
    return value.toLowerCase().includes(this.targetName.toLowerCase());
  }
}

export class ColumnFilterManager<T> {
  private strategies: FilterStrategy<T>[] = [];

  addStrategy(strategy: FilterStrategy<T>): void {
    this.strategies.push(strategy);
  }

  validate(value: any): boolean {
    return this.strategies.reduce<boolean>((acc, strategy) => {
      return acc && strategy.validate(value);
    }, true);
  }
}
