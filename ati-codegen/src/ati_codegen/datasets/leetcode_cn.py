from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class LeetCodeCnConfig:
    """
    leetcode.cn GraphQL 抓取配置。
    - cookie: 可选。遇到 403/风控时需要手动从浏览器复制 Cookie 贴进来。
    - csrf_token: 可选。有些请求需要 x-csrftoken（通常在 cookie 里也有）。
    """

    endpoint: str = "https://leetcode.cn/graphql/"
    cookie: str | None = None
    csrf_token: str | None = None
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    )
    timeout_s: int = 30
    sleep_s: float = 0.3
    max_retries: int = 5


class LeetCodeCnClient:
    def __init__(self, cfg: LeetCodeCnConfig):
        self.cfg = cfg
        self.sess = requests.Session()

    def _headers(self) -> dict[str, str]:
        h = {
            "Content-Type": "application/json",
            "User-Agent": self.cfg.user_agent,
            "Referer": "https://leetcode.cn/",
        }
        if self.cfg.cookie:
            h["Cookie"] = self.cfg.cookie
        if self.cfg.csrf_token:
            h["x-csrftoken"] = self.cfg.csrf_token
        return h

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_err: Exception | None = None
        for i in range(self.cfg.max_retries):
            try:
                resp = self.sess.post(
                    self.cfg.endpoint,
                    headers=self._headers(),
                    data=json.dumps(payload),
                    timeout=self.cfg.timeout_s,
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                if "errors" in data:
                    raise RuntimeError(f"GraphQL errors: {data['errors']}")
                return data["data"]
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(2**i, 10))
        raise RuntimeError(f"request failed after retries: {last_err!r}")

    def list_questions(self, skip: int, limit: int, difficulty: str | None = None) -> list[dict[str, Any]]:
        """
        返回题目列表（含 titleSlug、difficulty 等）。
        difficulty: "EASY"|"MEDIUM"|"HARD"|None
        """
        query = """
        query problemsetQuestionList($categorySlug: String, $limit: Int, $skip: Int, $filters: QuestionListFilterInput) {
          problemsetQuestionList: questionList(
            categorySlug: $categorySlug
            limit: $limit
            skip: $skip
            filters: $filters
          ) {
            total: totalNum
            questions: data {
              questionId
              frontendQuestionId: questionFrontendId
              title
              titleSlug
              difficulty
              paidOnly
              topicTags { name slug }
            }
          }
        }
        """
        filters: dict[str, Any] = {}
        if difficulty:
            filters["difficulty"] = difficulty
        payload = {"query": query, "variables": {"categorySlug": "", "skip": skip, "limit": limit, "filters": filters}}
        data = self._post(payload)
        return data["problemsetQuestionList"]["questions"]

    def fetch_question_detail(self, title_slug: str) -> dict[str, Any]:
        """
        拉取题目详情：内容、示例、代码骨架等。
        """
        query = """
        query questionData($titleSlug: String!) {
          question(titleSlug: $titleSlug) {
            questionId
            questionFrontendId
            title
            titleSlug
            content
            translatedContent
            difficulty
            sampleTestCase
            exampleTestcases
            codeSnippets { lang langSlug code }
            topicTags { name slug }
          }
        }
        """
        payload = {"query": query, "variables": {"titleSlug": title_slug}}
        data = self._post(payload)
        return data["question"]

