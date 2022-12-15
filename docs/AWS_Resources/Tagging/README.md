# Tagging AWS Resources

为了更好的管理AWS资源，你可以为每一个资源打tag。每一个tag都是一个标签，由用户定义的键和值组成。Tag可以帮助你管理、识别、组织、搜索和筛选资源。

## Best Practices

- Tag不要包含个人信息以及敏感信息
- Tag采用标准的驼峰式命名规则
- 考虑Tag支持多用途。例如：资源访问管理、成本跟踪、自动化以及组织形式
- 使用自动化工具来帮忙管理Tags。[AWS Resource Groups](https://docs.aws.amazon.com/ARG/latest/userguide/resource-groups.html) 和 [Resource Groups Tagging API](https://docs.aws.amazon.com/resourcegroupstagging/latest/APIReference/overview.html)支持程序化管理tags，使得用户可以更好的自动化的管理、搜索、过滤资源。
- 请记住，Tags应该很容易修改以适应不断变化的业务需求，但是需要考虑更改后带来的后果。例如，改变访问控制的tags时，你必须更新这些tags对应的policies。
- 通过使用AWS Organizations创建和部署tags策略，您可以自动执行组织选择采用的tags标准。

## Tags的分类

以**技术**、**业务**、**安全**的维度来最有效的使用tags以及创建tags groups

| Technical tags                                                                                                       | Tags for automation                                                                                                                            | Business tags                                                                                                                                    | Security tags       |   |    |            |     |           |          |      |                 |       |          |            |              |
|----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|---------------------|---|----|------------|-----|-----------|----------|------|-----------------|-------|----------|------------|--------------|
| **Name** – Identify individual resources                                                                             | **Date/Time** – Identify the date or time a resource should be started, stopped, deleted, or rotated                                           | **Project** – Identify projects that the resource supports                                                                                       | **Confidentiality** | – | An | identifier | for | the       | specific | data | confidentiality | level | a        | resource   | supports     |
| **Application ID** – Identify resources that are related to a specific Application                                   | **Opt in/Opt out** – Indicate whether a resource should be included in an automated activity such as starting, stopping, or resizing instances | **Owner** – Identify who is responsible for the resource                                                                                         | **Compliance**      | – | An | identifier | for | workloads | that     | must | adhere          | to    | specific | compliance | requirements |
| **Application Role** – Describe the function of a particular resource (such as web server, message broker, database) |                                                                                                                                                | **Cost Center/Business Unit** – Identify the cost center or business unit associated with a resource, typically for cost allocation and tracking |                     |   |    |            |     |           |          |      |                 |       |          |            |              |
| **Environment** – Distinguish between development, test, and production resources                                    |                                                                                                                                                |                                                                                                                                                  |                     |   |    |            |     |           |          |      |                 |       |          |            |              |


