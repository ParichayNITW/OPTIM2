import React from 'react';

interface Column<T> {
  key: keyof T;
  label: string;
  type?: 'number' | 'text';
  min?: number;
  step?: number;
}

interface EditableTableProps<T> {
  rows: T[];
  columns: Column<T>[];
  onChange: (rows: T[]) => void;
  minRows?: number;
  addLabel?: string;
  defaultRow: T;
}

export function EditableTable<T extends object>({
  rows, columns, onChange, minRows = 1, addLabel = '+ Add Row', defaultRow,
}: EditableTableProps<T>) {
  const update = (i: number, key: keyof T, val: unknown) => {
    const next = [...rows];
    next[i] = {...next[i], [key]: val};
    onChange(next);
  };
  const add = () => onChange([...rows, {...defaultRow}]);
  const remove = (i: number) => {
    if (rows.length <= minRows) return;
    onChange(rows.filter((_, idx) => idx !== i));
  };

  return (
    <div className="border border-gray-200 rounded-md overflow-x-auto text-sm">
      <table className="w-full">
        <thead className="bg-gray-50">
          <tr>
            {columns.map(c => (
              <th key={String(c.key)} className="px-2 py-1.5 text-left text-xs font-medium text-gray-600 border-b">
                {c.label}
              </th>
            ))}
            <th className="w-8 border-b" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b last:border-0 hover:bg-gray-50">
              {columns.map(c => (
                <td key={String(c.key)} className="px-1 py-0.5">
                  <input
                    type={c.type || 'number'}
                    value={String(row[c.key] ?? '')}
                    min={c.min}
                    step={c.step ?? 'any'}
                    onChange={e => update(i, c.key, c.type === 'text' ? e.target.value : parseFloat(e.target.value) || 0)}
                    className="w-full px-1 py-0.5 border border-transparent rounded focus:border-blue-400 focus:outline-none bg-transparent focus:bg-white text-xs"
                  />
                </td>
              ))}
              <td className="px-1 py-0.5 text-center">
                <button
                  onClick={() => remove(i)}
                  disabled={rows.length <= minRows}
                  className="text-gray-300 hover:text-red-500 disabled:cursor-not-allowed text-xs"
                  title="Remove row"
                >✕</button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="p-1">
        <button
          onClick={add}
          className="text-xs text-blue-600 hover:text-blue-800 font-medium px-2 py-0.5 rounded hover:bg-blue-50"
        >{addLabel}</button>
      </div>
    </div>
  );
}
