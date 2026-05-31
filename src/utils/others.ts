import dayjs from "dayjs";

const DATE_FORMATS = ["MMM D, YYYY", "MMM D", "YYYY/MM"] as const;

export const formatDate = (date: string | Date, type: 0 | 1 | 2 = 0) =>
  dayjs(date).format(DATE_FORMATS[type]);
