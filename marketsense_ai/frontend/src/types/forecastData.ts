export interface ForecastData {
  ds: string;
  yhat: string | number;
  yhat_lower: string | number;
  yhat_upper: string | number;
}